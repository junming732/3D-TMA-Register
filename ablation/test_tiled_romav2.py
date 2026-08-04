"""
Step 4 ablation — tiled RoMaV2 vs whole-image RoMaV2, isolated.
======================================================================
IMPORTANT — unlike steps 1-3, this one requires torch + the actual
`romav2` package + a GPU (your production dependencies). I don't have
those in my sandbox, so unlike the earlier scripts I could not run this
end-to-end or validate it against real RoMaV2 output before handing it
to you. What I *could* and did validate here: the tile-stitching math
(offset placement, confidence/edge-taper blending, background-zero and
displacement-cap applied once on the combined field) against a mocked
matcher with a known synthetic flow field, confirming the tiled path
reconstructs the same field as a single whole-image pass to within
floating-point tolerance. That check lives in a throwaway test, not in
this file — but it means the stitching arithmetic itself is sound; the
real network's behavior on real tissue is still yours to observe.
Please sanity-check the first few pairs' displacement-field visualizations
before trusting this on all 30 cores.

WHAT THIS ISOLATES: L0 is held at the current production baseline
(AKAZE + plain cv2.RANSAC — not yet whatever wins steps 1-2, swap
L0_METHOD below once that's decided). ROMA_MODE is fixed to 'ck_only'
(not swept across the other modes) so this is a clean single-variable
test of tiling alone, not tiling-interacting-with-channel-fusion.

HOW TILING WORKS: both fixed/moving CK-linear images (already affine-
prealigned by L0) are split into overlapping tiles. Each tile pair is
run through RoMaV2 independently — so each tile gets the model's full
800/1280 internal resolution budget applied to a much smaller physical
region than the whole image, at the cost of one model.match() call per
tile instead of one for the whole image. Per-tile predictions are
converted to GLOBAL pixel coordinates (tile offset added), then
blended: weight = (raised-cosine edge taper, only on edges with a real
neighboring tile) x (that tile's own confidence at that pixel). Where
combined weight is ~0 (no tile confidently covered that pixel),
falls back to identity. Background-zeroing (fixed tissue mask) and
displacement-cap (WARP_MAX_DISPLACEMENT_PX) are applied ONCE on the
final combined field — mirroring exactly what the whole-image path
does — so the two arms differ ONLY in how the raw prediction was
produced, not in post-processing.

Tiles with fixed-mask tissue coverage below TILE_MASK_MIN_FRAC are
skipped entirely (left as identity / no contribution) rather than
wasting a model call on background.

ALSO COMPUTES landmark-based TRE alongside NCC, reusing the L0 affine's
own inlier correspondences (matching VALIS's documented practice — no
second detector, no extra RoMaV2 calls). Because these are the same
points used to fit the affine, this measures whether each arm's warp
PRESERVES good alignment at already-resolved points — a stability/
non-regression check — not "how much did this arm improve over L0."
Arm-vs-arm TRE comparisons are still valid; comparing either arm against
the l0_only_tre sanity-check column is not (it's near-zero by
construction). See test_roma_mode_ablation.py's process_core for the
full reasoning (same design, first used there).

Usage:
    python test_tiled_romav2.py --start 1 --end 30
    python test_tiled_romav2.py --core_name coreA --tile_size 1536 --tile_overlap 256

Output:
    tiled_romav2_ablation.csv, written next to this script. Summary:
    ncc_warp win/tie/loss/mean-delta (tiled vs whole-image), coverage_pct
    and mean_confidence comparison, and mean wall-time per pair for each
    arm (tiling costs more model calls — this quantifies that cost).
"""

import os
import sys
import glob
import re
import argparse
import logging
import time
import gc

import numpy as np
import pandas as pd
import tifffile
import cv2
import SimpleITK as sitk
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
import config
from held_out_tre import compute_tre_warp, compute_tre_affine

os.environ['TORCH_HOME'] = os.path.join(config.DATASPACE, 'model_weights')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# Mask the GPU completely from this process before PyTorch initializes
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
torch._dynamo.config.disable = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants copied verbatim from akaze_romav2_multi_channel_warp.py — keep
# in sync manually if you retune the production config.
# ─────────────────────────────────────────────────────────────────────────────
AKAZE_THRESHOLD          = 0.0001
AKAZE_MAX_KEYPOINTS      = 20_000
LOWE_RATIO               = 0.80
MIN_MATCHES              = 20
MIN_INLIERS              = 6
RANSAC_CONFIDENCE        = 0.995
RANSAC_MAX_ITERS         = 5000
RANSAC_THRESH            = 8.0
MAX_SCALE_DEVIATION      = 0.08
MAX_SHEAR                = 0.15
MAX_ROTATION_DEG         = 15.0
L0_METHOD                = cv2.RANSAC  # swap once steps 1-2 pick a winner

ROMAV2_DEVICE            = 'cpu'
ROMAV2_H                 = 800
ROMAV2_W                 = 800
ROMAV2_H_HR              = 1280
ROMAV2_W_HR              = 1280
WARP_CONFIDENCE_THRESH   = 0.75
WARP_MAX_DISPLACEMENT_PX = 80.0
MASK_MIN_FRAC            = 0.05
MIN_CK_NONZERO_FRAC      = 0.01
CK_CHANNEL_IDX           = 6

# Step 4 — tiling defaults. Physical tile size in px (longer-side budget
# the model's internal 800/1280 grid gets applied to) and overlap width
# used for the blend taper. Override via --tile_size/--tile_overlap;
# these interact with your actual image dimensions so worth sweeping.
TILE_SIZE_DEFAULT    = 1536
TILE_OVERLAP_DEFAULT = 256
TILE_MASK_MIN_FRAC   = 0.05  # skip a tile if fixed-mask tissue coverage is below this

DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")
OUTPUT_CSV        = os.path.join(current_dir, "tre_tiled_romav2_ablation.csv")

# Held-out TRE (see held_out_tre.py) — reuses the L0 affine's own inlier
# correspondences as landmarks (matching VALIS's documented practice: no
# second detector, no extra RoMaV2 calls). Because these points also fit
# the affine, this measures whether each arm's warp PRESERVES good
# alignment at already-resolved points (a stability/non-regression check),
# not "how much did this arm improve over L0" — l0_only_tre_mean_px is a
# sanity check (expected near-zero), not a baseline to compare against.
# See test_roma_mode_ablation.py's process_core for the full reasoning.
# CHANGE THIS to your actual pixel size.
PIXEL_SIZE_UM = 0.497


# ─────────────────────────────────────────────────────────────────────────────
# Copied verbatim from production / the L0 harness
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


def prepare_ck(img_arr):
    """Returns (norm_lin, norm_log) uint8 — identical to production prepare_ck."""
    img_float = img_arr.astype(np.float32)
    p_lo_lin, p_hi_lin = np.percentile(img_float[::4, ::4], (0.1, 99.9))
    norm_lin = cv2.normalize(
        np.clip(img_float, p_lo_lin, p_hi_lin), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    log_img    = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log   = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return norm_lin, norm_log


def get_mask_sibling_path(tif_path):
    if tif_path.endswith(".ome.tif"):
        stem = tif_path[:-len(".ome.tif")]
    else:
        stem = os.path.splitext(tif_path)[0]
    return stem + "_tissue_mask.png"


def load_mask_or_none(tif_path, shape_hw):
    mask_path = get_mask_sibling_path(tif_path)
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == shape_hw:
            return mask
        logger.warning(f"Mask at {mask_path} missing/shape-mismatched — matching unmasked.")
    else:
        logger.warning(f"No precomputed mask for {os.path.basename(tif_path)} — matching unmasked.")
    return None


def constrain_affine(M):
    if M is None:
        return None
    M_out    = M.copy()
    U, S, Vt = np.linalg.svd(M_out[:2, :2])
    S        = np.clip(S, 1.0 - MAX_SCALE_DEVIATION, 1.0 + MAX_SCALE_DEVIATION)
    if S[1] > 1e-6 and S[0] / S[1] > 1.0 + MAX_SHEAR:
        S[0] = S[1] * (1.0 + MAX_SHEAR)
    M_out[:2, :2] = U @ np.diag(S) @ Vt
    return M_out


def transform_is_sane(M):
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R        = U @ Vt
    rot_deg  = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return rot_deg <= MAX_ROTATION_DEG


def measure_ncc(fixed_f32, moving_f32, mask_uint8):
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


def detect_and_match(fixed_log, moving_log, fixed_mask, moving_mask, slice_id):
    detector = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    kp1_raw, des1 = detector.detectAndCompute(fixed_log,  fixed_mask)
    kp2_raw, des2 = detector.detectAndCompute(moving_log, moving_mask)

    n1 = len(kp1_raw) if kp1_raw else 0
    n2 = len(kp2_raw) if kp2_raw else 0
    if des1 is None or des2 is None or n1 < 4 or n2 < 4:
        logger.warning(f"[{slice_id}] Feature starvation (fixed={n1}, moving={n2}).")
        return None, None, 0

    def cap_by_response(kps, des, max_kp):
        if len(kps) <= max_kp:
            return kps, des
        idx = np.argsort([kp.response for kp in kps])[::-1][:max_kp]
        return tuple(kps[i] for i in idx), des[idx]

    kp1, des1 = cap_by_response(kp1_raw, des1, AKAZE_MAX_KEYPOINTS)
    kp2, des2 = cap_by_response(kp2_raw, des2, AKAZE_MAX_KEYPOINTS)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw
              if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        logger.warning(f"[{slice_id}] Insufficient matches ({len(good)} < {MIN_MATCHES}).")
        return None, None, len(good)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return src_pts, dst_pts, len(good)


def fit_affine(src_pts, dst_pts, slice_id):
    """L0 baseline — plain RANSAC, current production config. Swap
    L0_METHOD at the top of the file once steps 1-2 pick a winner."""
    M, mask = cv2.estimateAffine2D(
        dst_pts, src_pts, method=L0_METHOD,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
    )
    if M is None or mask is None:
        logger.warning(f"[{slice_id}] L0 RANSAC diverged.")
        return None, 0, None
    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        logger.warning(f"[{slice_id}] L0 inlier count too low ({n_inliers}).")
        return None, n_inliers, None
    M = constrain_affine(M)
    if M is None or not transform_is_sane(M):
        logger.warning(f"[{slice_id}] L0 transform rejected (rotation out of range).")
        return None, n_inliers, None
    return M, n_inliers, mask


def to_rgb_pil(img):
    from PIL import Image
    if img.ndim == 2:
        return Image.fromarray(np.stack([img, img, img], axis=-1))
    elif img.ndim == 3 and img.shape[2] == 3:
        return Image.fromarray(img)
    else:
        raise ValueError(f"Unexpected image shape for RoMaV2: {img.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# L1: ROMAV2 MODEL (lazy singleton) — copied from production
# ─────────────────────────────────────────────────────────────────────────────

_romav2_model = None

def get_romav2_model():
    global _romav2_model
    if _romav2_model is None:
        from romav2 import RoMaV2
        logger.info("Loading RoMaV2 on strict CPU (GPU is masked)...")
        _romav2_model = RoMaV2().to('cpu')
        _romav2_model.eval()
        _romav2_model = torch._dynamo.disable(_romav2_model)
        _romav2_model.H_lr = ROMAV2_H
        _romav2_model.W_lr = ROMAV2_W
        _romav2_model.H_hr = ROMAV2_H_HR
        _romav2_model.W_hr = ROMAV2_W_HR
        logger.info("RoMaV2 model loaded.")
    return _romav2_model


def call_romav2_match(img_A_uint8, img_B_uint8):
    """
    Executes a purely CPU-bound match. 
    OOM caching logic removed; memory is handled entirely by system RAM.
    """
    model = get_romav2_model()
    img_A = to_rgb_pil(img_A_uint8)
    img_B = to_rgb_pil(img_B_uint8)
    
    try:
        with torch.no_grad():
            preds = model.match(img_A, img_B)
            
        warp_AB    = preds['warp_AB'].squeeze(0).cpu().numpy().copy()
        overlap_AB = preds['overlap_AB'].squeeze().cpu().numpy().copy()
        overlap_AB = overlap_AB.reshape(warp_AB.shape[0], warp_AB.shape[1])
        return warp_AB, overlap_AB
        
    finally:
        try:
            del preds
        except NameError:
            pass
        # Clear CPU memory garbage collection; torch.cuda.empty_cache() is no longer needed
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Whole-image RoMaV2 pass — copied/adapted from production romav2_dense_warp,
# used here as the baseline arm within THIS script (identical inputs to the
# tiled arm, for a fair comparison).
# ─────────────────────────────────────────────────────────────────────────────

def romav2_dense_warp_whole(fixed_lin, moving_lin, slice_id, orig_h, orig_w,
                            tissue_mask_full=None):
    try:
        warp_AB, overlap_AB = call_romav2_match(fixed_lin, moving_lin)
        H_lr, W_lr = warp_AB.shape[:2]

        b_coords_x = (warp_AB[..., 0] + 1.0) / 2.0 * (orig_w - 1)
        b_coords_y = (warp_AB[..., 1] + 1.0) / 2.0 * (orig_h - 1)

        confident_2d    = overlap_AB >= WARP_CONFIDENCE_THRESH
        n_confident     = int(confident_2d.sum())
        coverage_pct    = n_confident / (H_lr * W_lr) * 100
        mean_confidence = float(overlap_AB.mean())

        grid_x_lr  = np.linspace(0, orig_w - 1, W_lr, dtype=np.float32)
        grid_y_lr  = np.linspace(0, orig_h - 1, H_lr, dtype=np.float32)
        identity_x, identity_y = np.meshgrid(grid_x_lr, grid_y_lr)

        map_x_lr = np.where(confident_2d, b_coords_x, identity_x).astype(np.float32)
        map_y_lr = np.where(confident_2d, b_coords_y, identity_y).astype(np.float32)

        map_x = cv2.resize(map_x_lr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(map_y_lr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        map_x, map_y = _apply_cap_and_background(map_x, map_y, orig_h, orig_w, tissue_mask_full)

        return map_x, map_y, n_confident, coverage_pct, mean_confidence
    except Exception as exc:
        logger.warning(f"[{slice_id}] Whole-image RoMaV2 failed: {exc}")
        return None, None, 0, 0.0, 0.0


def _apply_cap_and_background(map_x, map_y, orig_h, orig_w, tissue_mask_full):
    """
    Shared final post-processing, applied identically regardless of
    whether map_x/map_y came from the whole-image or tiled path — so the
    two arms only differ in how the raw prediction was produced.
    """
    identity_x_full, identity_y_full = np.meshgrid(
        np.arange(orig_w, dtype=np.float32), np.arange(orig_h, dtype=np.float32))

    disp_x = map_x - identity_x_full
    disp_y = map_y - identity_y_full
    mag    = np.sqrt(disp_x**2 + disp_y**2)
    excess = mag > WARP_MAX_DISPLACEMENT_PX
    if np.any(excess):
        scale   = np.where(excess, WARP_MAX_DISPLACEMENT_PX / (mag + 1e-8), 1.0)
        disp_x *= scale
        disp_y *= scale
        map_x   = (identity_x_full + disp_x).astype(np.float32)
        map_y   = (identity_y_full + disp_y).astype(np.float32)

    if tissue_mask_full is not None:
        background = ~(tissue_mask_full.astype(bool))
        if np.any(background):
            map_x = map_x.copy(); map_y = map_y.copy()
            map_x[background] = identity_x_full[background]
            map_y[background] = identity_y_full[background]

    return map_x, map_y


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — tiled RoMaV2
# ─────────────────────────────────────────────────────────────────────────────

def compute_tile_grid(H, W, tile_size, overlap):
    """Sliding-window tile origins covering (H, W), overlap-stepped, with
    the last tile in each row/col shifted inward (not padded) to stay in
    bounds while keeping every tile the same physical size."""
    step = max(1, tile_size - overlap)

    def axis_origins(length):
        if length <= tile_size:
            return [0]
        origins = list(range(0, length - tile_size + 1, step))
        if origins[-1] != length - tile_size:
            origins.append(length - tile_size)
        return origins

    xs = axis_origins(W)
    ys = axis_origins(H)
    tiles = []
    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            tiles.append(dict(y0=y0, x0=x0, y1=y1, x1=x1,
                              xi=xi, yi=yi, n_cols=len(xs), n_rows=len(ys)))
    return tiles


def _raised_cosine_taper_1d(length, overlap, has_prev, has_next):
    w = np.ones(length, dtype=np.float32)
    if overlap <= 0:
        return w
    n = min(overlap, length // 2 if length // 2 > 0 else length)
    if has_prev and n > 0:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(np.float32)
        w[:n] *= ramp
    if has_next and n > 0:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(np.float32)[::-1]
        w[-n:] *= ramp
    return w


def tile_edge_taper(th, tw, overlap, tile):
    wx = _raised_cosine_taper_1d(tw, overlap, tile["xi"] > 0, tile["xi"] < tile["n_cols"] - 1)
    wy = _raised_cosine_taper_1d(th, overlap, tile["yi"] > 0, tile["yi"] < tile["n_rows"] - 1)
    return np.outer(wy, wx).astype(np.float32)


def romav2_match_tile_raw(fixed_tile, moving_tile, tile, slice_id):
    """
    One tile's raw prediction, placed into GLOBAL coordinates (tile
    offset added on the moving side). Returns (map_x_global, map_y_global,
    confidence) each shaped (th, tw), or None if the model call fails.
    No background-zeroing or displacement-cap here — that happens once,
    globally, after all tiles are blended (see _apply_cap_and_background).
    """
    th = tile["y1"] - tile["y0"]
    tw = tile["x1"] - tile["x0"]
    try:
        warp_AB, overlap_AB = call_romav2_match(fixed_tile, moving_tile)
        H_lr, W_lr = warp_AB.shape[:2]

        # tile-local moving-pixel coords
        b_coords_x = (warp_AB[..., 0] + 1.0) / 2.0 * (tw - 1)
        b_coords_y = (warp_AB[..., 1] + 1.0) / 2.0 * (th - 1)

        map_x_lr = cv2.resize(b_coords_x.astype(np.float32), (tw, th), interpolation=cv2.INTER_CUBIC)
        map_y_lr = cv2.resize(b_coords_y.astype(np.float32), (tw, th), interpolation=cv2.INTER_CUBIC)
        conf_full = cv2.resize(overlap_AB.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
        conf_full = np.clip(conf_full, 0.0, 1.0)

        # tile-local -> global: add this tile's origin on the moving side
        map_x_global = map_x_lr + tile["x0"]
        map_y_global = map_y_lr + tile["y0"]

        return map_x_global, map_y_global, conf_full
    except Exception as exc:
        logger.warning(f"[{slice_id}] Tile ({tile['y0']},{tile['x0']}) RoMaV2 call failed: {exc}")
        return None


def romav2_dense_warp_tiled(fixed_lin, moving_lin, slice_id, orig_h, orig_w,
                            tissue_mask_full=None,
                            tile_size=TILE_SIZE_DEFAULT, tile_overlap=TILE_OVERLAP_DEFAULT):
    """
    Tiled counterpart to romav2_dense_warp_whole, same signature/return
    shape, so process_core can call either interchangeably.
    """
    tiles = compute_tile_grid(orig_h, orig_w, tile_size, tile_overlap)
    logger.info(f"[{slice_id}] Tiled RoMaV2: {len(tiles)} tile(s) "
               f"({tile_size}px, overlap={tile_overlap}px)")

    acc_x = np.zeros((orig_h, orig_w), dtype=np.float64)
    acc_y = np.zeros((orig_h, orig_w), dtype=np.float64)
    acc_w = np.zeros((orig_h, orig_w), dtype=np.float64)
    conf_max = np.zeros((orig_h, orig_w), dtype=np.float32)

    n_tiles_run = 0
    n_tiles_skipped_tissue = 0
    n_tiles_failed = 0

    for tile in tiles:
        y0, y1, x0, x1 = tile["y0"], tile["y1"], tile["x0"], tile["x1"]

        if tissue_mask_full is not None:
            mask_crop = tissue_mask_full[y0:y1, x0:x1]
            tissue_frac = np.count_nonzero(mask_crop) / mask_crop.size
            if tissue_frac < TILE_MASK_MIN_FRAC:
                n_tiles_skipped_tissue += 1
                continue

        fixed_crop  = fixed_lin[y0:y1, x0:x1]
        moving_crop = moving_lin[y0:y1, x0:x1]

        result = romav2_match_tile_raw(fixed_crop, moving_crop, tile, slice_id)
        if result is None:
            n_tiles_failed += 1
            continue
        map_x_g, map_y_g, conf = result
        n_tiles_run += 1

        taper  = tile_edge_taper(y1 - y0, x1 - x0, tile_overlap, tile)
        weight = taper * conf

        acc_x[y0:y1, x0:x1]    += weight * map_x_g
        acc_y[y0:y1, x0:x1]    += weight * map_y_g
        acc_w[y0:y1, x0:x1]    += weight
        conf_max[y0:y1, x0:x1]  = np.maximum(conf_max[y0:y1, x0:x1], conf)

    logger.info(f"[{slice_id}] Tiled RoMaV2: {n_tiles_run} run, "
               f"{n_tiles_skipped_tissue} skipped (low tissue), "
               f"{n_tiles_failed} failed.")

    if n_tiles_run == 0:
        return None, None, 0, 0.0, 0.0

    identity_x_full, identity_y_full = np.meshgrid(
        np.arange(orig_w, dtype=np.float32), np.arange(orig_h, dtype=np.float32))

    valid  = acc_w > 1e-6
    map_x  = np.where(valid, acc_x / np.maximum(acc_w, 1e-6), identity_x_full).astype(np.float32)
    map_y  = np.where(valid, acc_y / np.maximum(acc_w, 1e-6), identity_y_full).astype(np.float32)

    n_confident     = int((conf_max >= WARP_CONFIDENCE_THRESH).sum())
    coverage_pct    = n_confident / (orig_h * orig_w) * 100
    mean_confidence = float(conf_max.mean())

    map_x, map_y = _apply_cap_and_background(map_x, map_y, orig_h, orig_w, tissue_mask_full)

    return map_x, map_y, n_confident, coverage_pct, mean_confidence


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_ck_pair(file_a, file_b):
    ck_a = tifffile.imread(file_a, key=CK_CHANNEL_IDX).astype(np.float32)
    ck_b = tifffile.imread(file_b, key=CK_CHANNEL_IDX).astype(np.float32)
    return ck_a, ck_b


def process_core(core_name, rows, tile_size, tile_overlap):
    input_dir = os.path.join(DATA_BASE_PATH, core_name)
    if not os.path.exists(input_dir):
        logger.error(f"[{core_name}] Input folder not found: {input_dir}")
        return

    sample_files = sorted(
        glob.glob(os.path.join(input_dir, "*.tif")) +
        glob.glob(os.path.join(input_dir, "*.tiff")),
        key=get_slice_number
    )
    file_list = [f for f in sample_files if "_thumb" not in os.path.basename(f)]

    allowed = load_slice_filter(SLICE_FILTER_YAML, core_name)
    if allowed is not None:
        file_list = [f for f in file_list if get_slice_number(f) in allowed]

    if len(file_list) < 2:
        logger.error(f"[{core_name}] Fewer than 2 slices after filtering — skipping.")
        return

    logger.info(f"[{core_name}] {len(file_list)} slices — {len(file_list)-1} adjacent pairs.")

    for i in range(len(file_list) - 1):
        file_a, file_b = file_list[i], file_list[i + 1]
        sid = f"{core_name}_Z{i:03d}-Z{i+1:03d}"

        ck_fixed, ck_moving = load_ck_pair(file_a, file_b)
        h, w = ck_fixed.shape
        fixed_mask  = load_mask_or_none(file_a, (h, w))
        moving_mask = load_mask_or_none(file_b, (h, w))

        fixed_lin,  fixed_log  = prepare_ck(ck_fixed)
        moving_lin, moving_log = prepare_ck(ck_moving)

        tissue_frac = (np.count_nonzero(fixed_mask) / fixed_mask.size
                      if fixed_mask is not None else 1.0)
        if tissue_frac < MASK_MIN_FRAC:
            logger.info(f"[{sid}] Tissue fraction {tissue_frac:.3f} < {MASK_MIN_FRAC} — skipping.")
            continue

        # ── L0: AKAZE + RANSAC affine (current production baseline) ──
        src_pts, dst_pts, n_matches = detect_and_match(
            fixed_log, moving_log, fixed_mask, moving_mask, sid)
        if src_pts is None:
            logger.info(f"[{sid}] L0 failed — skipping (step 4 needs an affine-prealigned pair).")
            continue
        M, n_inliers, inlier_mask = fit_affine(src_pts, dst_pts, sid)
        if M is None:
            logger.info(f"[{sid}] L0 affine rejected — skipping.")
            continue

        moving_lin_affine = cv2.warpAffine(
            moving_lin, M, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        moving_log_affine = cv2.warpAffine(
            moving_log, M, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        ncc_affine = measure_ncc(fixed_log.astype(np.float32),
                                 moving_log_affine.astype(np.float32), fixed_mask)

        row = dict(core=core_name, pair=sid, n_matches=n_matches,
                  n_inliers=n_inliers, ncc_affine=ncc_affine)

        # ── Held-out TRE setup — reuses L0's own inlier correspondences as
        # landmarks (see PIXEL_SIZE_UM comment above for the full reasoning:
        # this is a stability check, not an improvement measure). ──
        landmarks_fixed  = src_pts[inlier_mask.ravel() == 1].reshape(-1, 2)
        landmarks_moving = dst_pts[inlier_mask.ravel() == 1].reshape(-1, 2)
        image_diag_px = float(np.hypot(h, w))
        ground_truth_targets = (M[:, :2] @ landmarks_moving.T).T + M[:, 2]
        row["tre_n_landmarks"] = len(landmarks_fixed)
        l0_tre = compute_tre_affine(M, landmarks_fixed, landmarks_moving,
                                    px_size_um=PIXEL_SIZE_UM, image_diag_px=image_diag_px)
        row["l0_only_tre_mean_px"]   = l0_tre["mean_px"]  # sanity check, not a baseline
        row["l0_only_tre_median_px"] = l0_tre["median_px"]

        # ── L1a: whole-image RoMaV2 ──
        t0 = time.time()
        map_x_w, map_y_w, nconf_w, cov_w, meanconf_w = romav2_dense_warp_whole(
            fixed_lin, moving_lin_affine, sid, h, w, tissue_mask_full=fixed_mask)
        time_whole = time.time() - t0

        row["whole_time_s"]       = time_whole
        row["whole_coverage_pct"] = cov_w
        row["whole_mean_conf"]    = meanconf_w
        if map_x_w is not None:
            warped_ck = cv2.remap(moving_log_affine.astype(np.float32), map_x_w, map_y_w,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            row["whole_ncc_warp"] = measure_ncc(fixed_log.astype(np.float32), warped_ck, fixed_mask)
            tre_w = compute_tre_warp(map_x_w, map_y_w, landmarks_fixed, ground_truth_targets,
                                     px_size_um=PIXEL_SIZE_UM, image_diag_px=image_diag_px)
            row["whole_tre_mean_px"] = tre_w["mean_px"]
            row["whole_tre_median_px"] = tre_w["median_px"]
        else:
            row["whole_ncc_warp"] = np.nan
            row["whole_tre_mean_px"] = np.nan
            row["whole_tre_median_px"] = np.nan

        # ── L1b: tiled RoMaV2 ──
        t0 = time.time()
        map_x_t, map_y_t, nconf_t, cov_t, meanconf_t = romav2_dense_warp_tiled(
            fixed_lin, moving_lin_affine, sid, h, w, tissue_mask_full=fixed_mask,
            tile_size=tile_size, tile_overlap=tile_overlap)
        time_tiled = time.time() - t0

        row["tiled_time_s"]       = time_tiled
        row["tiled_coverage_pct"] = cov_t
        row["tiled_mean_conf"]    = meanconf_t
        if map_x_t is not None:
            warped_ck_t = cv2.remap(moving_log_affine.astype(np.float32), map_x_t, map_y_t,
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            row["tiled_ncc_warp"] = measure_ncc(fixed_log.astype(np.float32), warped_ck_t, fixed_mask)
            tre_t = compute_tre_warp(map_x_t, map_y_t, landmarks_fixed, ground_truth_targets,
                                     px_size_um=PIXEL_SIZE_UM, image_diag_px=image_diag_px)
            row["tiled_tre_mean_px"] = tre_t["mean_px"]
            row["tiled_tre_median_px"] = tre_t["median_px"]
        else:
            row["tiled_ncc_warp"] = np.nan
            row["tiled_tre_mean_px"] = np.nan
            row["tiled_tre_median_px"] = np.nan

        rows.append(row)
        logger.info(
            f"[{sid}] ncc_affine={ncc_affine:.4f} l0_only_tre={row['l0_only_tre_mean_px']:.2f}px "
            f"(median={row['l0_only_tre_median_px']:.2f}px) | "
            f"whole: ncc={row['whole_ncc_warp']:.4f} tre_mean={row['whole_tre_mean_px']:.2f}px "
            f"tre_median={row['whole_tre_median_px']:.2f}px cov={cov_w:.1f}% t={time_whole:.1f}s | "
            f"tiled: ncc={row['tiled_ncc_warp']:.4f} tre_mean={row['tiled_tre_mean_px']:.2f}px "
            f"tre_median={row['tiled_tre_median_px']:.2f}px cov={cov_t:.1f}% t={time_tiled:.1f}s"
        )


def summarize(df):
    print("\n" + "=" * 70)
    print("STEP 4 — Tiled RoMaV2 vs whole-image RoMaV2 (ncc_warp, more negative = better)")
    print("=" * 70)
    valid = df.dropna(subset=["whole_ncc_warp", "tiled_ncc_warp"])
    print(f"Pairs total: {len(df)} | both arms succeeding: {len(valid)}")
    if len(valid) == 0:
        print("No pairs with both arms succeeding — cannot compare.")
        return

    delta = valid["tiled_ncc_warp"] - valid["whole_ncc_warp"]  # negative = tiled better
    n_better = int((delta < -1e-6).sum())
    n_worse  = int((delta > 1e-6).sum())
    n_tied   = len(valid) - n_better - n_worse
    print(f"Tiled better: {n_better} | worse: {n_worse} | tied: {n_tied}")
    print(f"Mean delta (tiled - whole): {delta.mean():.5f} (negative = tiled better)")
    print(f"Median delta:               {delta.median():.5f}")

    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(valid["whole_ncc_warp"], valid["tiled_ncc_warp"])
        print(f"Wilcoxon signed-rank p-value: {p:.4f}")
    except ImportError:
        pass

    print(f"\nMean coverage_pct — whole: {valid['whole_coverage_pct'].mean():.1f}% | "
         f"tiled: {valid['tiled_coverage_pct'].mean():.1f}%")
    print(f"Mean confidence   — whole: {valid['whole_mean_conf'].mean():.3f} | "
         f"tiled: {valid['tiled_mean_conf'].mean():.3f}")

    mean_t_whole = df["whole_time_s"].mean()
    mean_t_tiled = df["tiled_time_s"].mean()
    print(f"\nMean wall-time per pair — whole: {mean_t_whole:.1f}s | "
         f"tiled: {mean_t_tiled:.1f}s | tiled is {mean_t_tiled/mean_t_whole:.1f}x the cost")

    whole_failed = df[df["whole_ncc_warp"].isna()]
    if len(whole_failed) > 0:
        rescued = whole_failed["tiled_ncc_warp"].notna().sum()
        print(f"\nWhole-image failed on {len(whole_failed)} pair(s); "
             f"tiled succeeded on {rescued} of those.")
    tiled_failed = df[df["tiled_ncc_warp"].isna()]
    if len(tiled_failed) > 0:
        rescued_by_whole = tiled_failed["whole_ncc_warp"].notna().sum()
        print(f"Tiled failed on {len(tiled_failed)} pair(s); "
             f"whole-image succeeded on {rescued_by_whole} of those.")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # SAME COMPARISON, BUT BY LANDMARK STABILITY INSTEAD OF NCC
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAME COMPARISON, BUT BY LANDMARK STABILITY INSTEAD OF NCC (px, lower = better)")
    print("Reuses the L0 affine's own inlier correspondences (matching VALIS's")
    print("documented practice), sampling the SAME deformation fields used above.")
    print("NOTE: measures whether each arm's warp PRESERVES good alignment at")
    print("already-resolved points, not 'how much did this arm improve over L0' —")
    print("l0_only_tre_mean_px is a sanity check (expected near-zero), not a baseline.")
    print("=" * 70)

    if "l0_only_tre_mean_px" in df.columns:
        l0_valid = df["l0_only_tre_mean_px"].dropna()
        print(f"L0-only sanity check: n={len(l0_valid)} mean={l0_valid.mean():.3f}px "
             f"(should be small — large values here mean something's wrong upstream)")

    tre_valid = df.dropna(subset=["whole_tre_mean_px", "tiled_tre_mean_px"])
    print(f"\nPairs with both arms' TRE available: {len(tre_valid)}")
    if len(tre_valid) > 0:
        tre_delta = tre_valid["tiled_tre_mean_px"] - tre_valid["whole_tre_mean_px"]
        tre_better = int((tre_delta < -1e-6).sum())
        tre_worse  = int((tre_delta > 1e-6).sum())
        tre_tied   = len(tre_valid) - tre_better - tre_worse
        print(f"Tiled lower TRE (better): {tre_better} | higher TRE (worse): {tre_worse} | "
             f"tied: {tre_tied}")
        print(f"Mean TRE — whole: {tre_valid['whole_tre_mean_px'].mean():.3f}px | "
             f"tiled: {tre_valid['tiled_tre_mean_px'].mean():.3f}px")
        p_tre = np.nan
        try:
            from scipy.stats import wilcoxon
            _, p_tre = wilcoxon(tre_valid["whole_tre_mean_px"], tre_valid["tiled_tre_mean_px"])
            print(f"Wilcoxon signed-rank p-value (TRE): {p_tre:.4f}")
        except ImportError:
            pass

        ncc_says_tiled_better = n_better > n_worse
        tre_says_tiled_better = tre_better > tre_worse
        if ncc_says_tiled_better == tre_says_tiled_better:
            print(">>> NCC and landmark-stability AGREE on direction.")
        else:
            print(">>> NCC and landmark-stability DISAGREE on direction — worth a closer look "
                 "before trusting either verdict alone.")
    else:
        print("No pairs with usable TRE for both arms.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Isolated tiled vs whole-image RoMaV2 ablation.")
    parser.add_argument("--core_name", type=str, action="append", default=[])
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--tile_size", type=int, default=TILE_SIZE_DEFAULT,
                        help=f"Physical tile size in px. Default: {TILE_SIZE_DEFAULT}.")
    parser.add_argument("--tile_overlap", type=int, default=TILE_OVERLAP_DEFAULT,
                        help=f"Overlap width in px for blend taper. Default: {TILE_OVERLAP_DEFAULT}.")
    args = parser.parse_args()

    core_names = list(args.core_name)
    if args.start is not None and args.end is not None:
        core_names += [f"Core_{i:02d}" for i in range(args.start, args.end + 1)]
    if not core_names:
        parser.error("Provide at least one --core_name, or both --start and --end.")

    total  = len(core_names)
    rows   = []
    status = {}

    logger.info("=" * 60)
    logger.info(f"Tiled vs whole-image RoMaV2 ablation — {total} core(s), "
               f"tile_size={args.tile_size}, tile_overlap={args.tile_overlap}")
    logger.info("=" * 60)

    for idx, core_name in enumerate(core_names, start=1):
        logger.info(f"[{idx}/{total}] {core_name}")
        n_before = len(rows)
        try:
            process_core(core_name, rows, args.tile_size, args.tile_overlap)
            status[core_name] = "OK" if len(rows) > n_before else "NO_PAIRS"
        except Exception as exc:
            logger.error(f"[{core_name}] Crashed — skipping, continuing batch: {exc}")
            status[core_name] = "CRASHED"

    logger.info("-" * 60)
    for core_name in core_names:
        logger.info(f"    {core_name:<14} {status.get(core_name, 'UNKNOWN')}")
    logger.info("-" * 60)

    if not rows:
        logger.error("No pairs processed across any core — nothing to write.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Wrote {len(df)} rows → {OUTPUT_CSV}")

    summarize(df)


if __name__ == "__main__":
    main()