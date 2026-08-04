"""
Step 5 ablation — ROMA_MODE sweep (which image RoMaV2 sees), isolated.
======================================================================
Same caveat as step 4: this needs torch + your `romav2` package + a GPU,
none of which exist in my sandbox, so I couldn't run the RoMaV2 call
itself here. What I DID validate here (pure numpy/opencv/skimage, no
torch needed): the five per-mode compositing functions
(prepare_ck / prepare_3ch_fusion / prepare_color_lut_fusion /
clahe_normalize) against synthetic multi-channel data — correct output
shape (grayscale vs RGB), dtype (uint8), and value range [0, 255] for
each mode. That confirms the plumbing that builds each mode's image is
correct; the RoMaV2 matching quality on real tissue is still yours to
observe.

WHAT THIS ISOLATES:
- L0 is a SINGLE run per pair, always on CK (matches production: AKAZE
  never depends on ROMA_MODE), using the current baseline solver
  (L0_METHOD, swap once steps 1-2 pick a winner). One affine M per pair,
  shared across every mode's arm below — so all modes start from an
  identical pre-alignment, and any difference downstream is attributable
  to ROMA_MODE alone.
- L1 is WHOLE-IMAGE RoMaV2 (not tiled) for every mode — tiling is a
  separate open variable (step 4); combining it with a ROMA_MODE sweep
  here would confound which change caused what. Revisit this script
  with tiling once step 4 has an answer, if useful.
- The evaluation metric is ALWAYS ncc_warp on CK-log, for every mode,
  including modes that feed RoMaV2 something else entirely (DAPI,
  fusion composites). This mirrors your production gate exactly, and
  is deliberate: the question isn't "how well did RoMaV2 align the
  image it was looking at", it's "how well did the resulting transform
  align the channel you actually care about downstream." A mode that
  matches beautifully on DAPI but doesn't transfer to CK is not a win.

Modes tested: ck_only, 3ch_fusion, color_lut, dapi_clahe, ck_clahe — the
same five your production script supports. ck_only is treated as the
reference/baseline in the summary since it's the original default
behavior; the others are candidates being tested against it.

ALSO COMPUTES landmark-based TRE alongside NCC, reusing the L0 affine's
own inlier correspondences (matching VALIS's documented practice — no
second detector, no extra RoMaV2 calls). Because these are the same
points used to fit the affine, this measures whether each mode's warp
PRESERVES good alignment at already-resolved points — a stability/
non-regression check — not "how much did this mode improve over L0."
Mode-vs-mode TRE comparisons are still valid; comparing any mode against
the l0_only_tre sanity-check column is not (it's near-zero by
construction). See the inline comments in process_core for the full
reasoning.

Usage:
    python test_roma_mode_ablation.py --start 1 --end 30
    python test_roma_mode_ablation.py --core_name coreA --modes ck_only,dapi_clahe

Output:
    roma_mode_ablation.csv, written next to this script (one row per
    pair, one ncc_warp/coverage/confidence/time column-set per mode).
    Summary: per-mode mean ncc_warp + win/tie/loss vs ck_only + Wilcoxon
    p-value, an overall ranking by mean ncc_warp, and (if enough pairs
    have every mode succeed) a Friedman test across all modes at once.
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
# Constants copied verbatim from akaze_romav2_multi_channel_warp.py
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

DAPI_CHANNEL_IDX = 0
CK_CHANNEL_IDX   = 6
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

ALL_MODES = ['ck_only', '3ch_fusion', 'color_lut', 'dapi_clahe', 'ck_clahe']

DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")
OUTPUT_CSV        = os.path.join(current_dir, "tre_roma_mode_ablation.csv")

# Held-out TRE (see held_out_tre.py) — reuses the L0 affine's own inlier
# correspondences as landmarks (matching VALIS's documented practice: reuse
# rigid-stage matched features, filtered for reliability, to evaluate the
# pipeline downstream of that stage). No second detector, no extra compute:
# the SAME deformation field computed for the ncc_warp comparison below is
# just sampled at these points too. Because these points were also used to
# fit the affine, this measures whether each mode's warp PRESERVES good
# alignment at already-resolved points (a stability check), not "how much
# did L1 improve over L0" — see the inline note in process_core for why.
# CHANGE THIS to your actual pixel size — TRE in microns is meaningless
# until it reflects your real imaging resolution, not a placeholder.
PIXEL_SIZE_UM = 0.497


# ─────────────────────────────────────────────────────────────────────────────
# Copied verbatim from production / earlier harnesses
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


def clahe_normalize(img_arr):
    from skimage import exposure
    img_float = img_arr.astype(np.float32)
    img01 = exposure.rescale_intensity(img_float, out_range=(0, 1))
    eq    = exposure.equalize_adapthist(img01)
    return exposure.rescale_intensity(eq, out_range=(0, 255)).astype(np.uint8)


def _prepare_single(img_arr, lo=0.1, hi=99.5):
    img_float = img_arr.astype(np.float32)
    log_img   = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (lo, hi))
    return cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)


def prepare_3ch_fusion(vol):
    dapi = _prepare_single(vol[DAPI_CHANNEL_IDX].astype(np.float32))
    af   = _prepare_single(vol[AF_CHANNEL_IDX].astype(np.float32))
    ck   = _prepare_single(vol[CK_CHANNEL_IDX].astype(np.float32))
    return np.stack([dapi, af, ck], axis=-1)  # (H, W, 3)


def prepare_color_lut_fusion(vol):
    h, w = vol.shape[1], vol.shape[2]
    acc  = np.zeros((h, w, 3), dtype=np.float32)
    n    = len(COLOR_LUT)
    for idx, color in COLOR_LUT.items():
        norm      = _prepare_single(vol[idx].astype(np.float32)).astype(np.float32) / 255.0
        color_arr = np.array(color, dtype=np.float32) / 255.0
        acc      += norm[..., None] * color_arr[None, None, :]
    return np.clip(acc / n * 255.0, 0, 255).astype(np.uint8)


def build_roma_input(mode, vol):
    """vol: (C, H, W) float/uint array, full multi-channel slice."""
    if mode == '3ch_fusion':
        return prepare_3ch_fusion(vol)
    elif mode == 'color_lut':
        return prepare_color_lut_fusion(vol)
    elif mode == 'dapi_clahe':
        return clahe_normalize(vol[DAPI_CHANNEL_IDX].astype(np.float32))
    elif mode == 'ck_clahe':
        return clahe_normalize(vol[CK_CHANNEL_IDX].astype(np.float32))
    else:  # 'ck_only'
        lin, _ = prepare_ck(vol[CK_CHANNEL_IDX].astype(np.float32))
        return lin


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
# L1: ROMAV2 MODEL (Strict CPU)
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


def log_gpu_mem(tag):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"[{tag}] GPU mem — allocated={alloc:.2f} GiB reserved={reserved:.2f} GiB "
                   f"(this process only; other processes on the GPU aren't visible here)")


def _apply_cap_and_background(map_x, map_y, orig_h, orig_w, tissue_mask_full):
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


def romav2_dense_warp_whole(fixed_input, moving_input, slice_id, orig_h, orig_w,
                            tissue_mask_full=None):
    try:
        warp_AB, overlap_AB = call_romav2_match(fixed_input, moving_input)
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

        return map_x, map_y, coverage_pct, mean_confidence
    except Exception as exc:
        logger.warning(f"[{slice_id}] RoMaV2 failed: {exc}")
        return None, None, 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_full_volume(file_path):
    """Full (C, H, W) multi-channel array — needed here since different
    modes need different channel subsets (DAPI, AF, CK, or all 7)."""
    return tifffile.imread(file_path).astype(np.float32)


def process_core(core_name, rows, modes):
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

        vol_fixed  = load_full_volume(file_a)
        vol_moving = load_full_volume(file_b)
        h, w = vol_fixed.shape[1], vol_fixed.shape[2]

        fixed_mask  = load_mask_or_none(file_a, (h, w))
        moving_mask = load_mask_or_none(file_b, (h, w))

        tissue_frac = (np.count_nonzero(fixed_mask) / fixed_mask.size
                      if fixed_mask is not None else 1.0)
        if tissue_frac < MASK_MIN_FRAC:
            logger.info(f"[{sid}] Tissue fraction {tissue_frac:.3f} < {MASK_MIN_FRAC} — skipping.")
            continue

        # ── L0: single AKAZE+RANSAC affine on CK, shared by every mode ──
        fixed_lin,  fixed_log  = prepare_ck(vol_fixed[CK_CHANNEL_IDX])
        moving_lin, moving_log = prepare_ck(vol_moving[CK_CHANNEL_IDX])

        src_pts, dst_pts, n_matches = detect_and_match(
            fixed_log, moving_log, fixed_mask, moving_mask, sid)
        if src_pts is None:
            logger.info(f"[{sid}] L0 failed — skipping (need a shared affine for all modes).")
            continue
        M, n_inliers, inlier_mask = fit_affine(src_pts, dst_pts, sid)
        if M is None:
            logger.info(f"[{sid}] L0 affine rejected — skipping.")
            continue

        moving_log_affine = cv2.warpAffine(
            moving_log, M, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ncc_affine = measure_ncc(fixed_log.astype(np.float32),
                                 moving_log_affine.astype(np.float32), fixed_mask)

        row = dict(core=core_name, pair=sid, n_matches=n_matches,
                  n_inliers=n_inliers, ncc_affine=ncc_affine)

        # ── Held-out TRE setup — reuses the L0 affine's OWN inlier
        # correspondences as landmarks (matching VALIS's actual documented
        # practice: reuse rigid-stage matched features, filtered for
        # reliability, to evaluate the pipeline downstream of that stage).
        # No second detector, no extra compute — the mask is already sitting
        # in memory from the fit_affine call above.
        #
        # IMPORTANT INTERPRETATION NOTE: because these are the SAME points
        # used to fit M, ground_truth_targets below is trivially close to
        # each point's own fixed-space position (that's what "inlier" means
        # — small residual is exactly what the solver optimized for). So
        # this does NOT measure "how much did L1 improve over L0" — an
        # inlier-only l0_only_tre would be near-zero by construction, not a
        # meaningful baseline. What it DOES measure: whether each mode's
        # dense warp PRESERVES good alignment at points L0 already resolved
        # well, i.e. a stability/non-regression check — a warp that
        # introduces spurious local distortion even in well-aligned regions
        # will show up here as elevated TRE relative to other modes, even
        # though none of them are being compared against a true "before"
        # baseline. Mode-vs-mode comparisons are still valid and meaningful;
        # comparing any mode against l0_only_tre is not.
        landmarks_fixed  = src_pts[inlier_mask.ravel() == 1].reshape(-1, 2)
        landmarks_moving = dst_pts[inlier_mask.ravel() == 1].reshape(-1, 2)
        image_diag_px = float(np.hypot(h, w))

        ground_truth_targets = (M[:, :2] @ landmarks_moving.T).T + M[:, 2]
        row["tre_n_landmarks"] = len(landmarks_fixed)
        # Sanity-check column, not a baseline — expected to be near-zero by
        # construction (see note above). A NON-trivial value here would
        # actually indicate something is wrong (M and the inlier mask
        # disagreeing), so it's worth keeping as a consistency check.
        l0_tre = compute_tre_affine(M, landmarks_fixed, landmarks_moving,
                                    px_size_um=PIXEL_SIZE_UM, image_diag_px=image_diag_px)
        row["l0_only_tre_mean_px"]   = l0_tre["mean_px"]
        row["l0_only_tre_median_px"] = l0_tre["median_px"]

        # ── L1: one whole-image RoMaV2 pass per mode, same M, same eval channel ──
        log_gpu_mem(sid)
        for mode in modes:
            fixed_input  = build_roma_input(mode, vol_fixed)
            moving_input = build_roma_input(mode, vol_moving)
            moving_input_affine = cv2.warpAffine(
                moving_input, M, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            t0 = time.time()
            map_x, map_y, cov, meanconf = romav2_dense_warp_whole(
                fixed_input, moving_input_affine, f"{sid}_{mode}", h, w,
                tissue_mask_full=fixed_mask)
            elapsed = time.time() - t0

            row[f"{mode}_time_s"]       = elapsed
            row[f"{mode}_coverage_pct"] = cov
            row[f"{mode}_mean_conf"]    = meanconf
            if map_x is not None:
                warped_ck = cv2.remap(moving_log_affine.astype(np.float32), map_x, map_y,
                                      interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                row[f"{mode}_ncc_warp"] = measure_ncc(
                    fixed_log.astype(np.float32), warped_ck, fixed_mask)

                tre = compute_tre_warp(map_x, map_y, landmarks_fixed, ground_truth_targets,
                                       px_size_um=PIXEL_SIZE_UM, image_diag_px=image_diag_px)
                row[f"{mode}_tre_mean_px"]   = tre["mean_px"]
                row[f"{mode}_tre_median_px"] = tre["median_px"]
            else:
                row[f"{mode}_ncc_warp"] = np.nan
                row[f"{mode}_tre_mean_px"] = np.nan
                row[f"{mode}_tre_median_px"] = np.nan

        rows.append(row)
        summary_str = " | ".join(
            f"{m}: ncc={row.get(f'{m}_ncc_warp', float('nan')):.4f} "
            f"tre={row.get(f'{m}_tre_mean_px', float('nan')):.2f}px" for m in modes)
        logger.info(f"[{sid}] ncc_affine={ncc_affine:.4f} "
                   f"l0_only_tre={row['l0_only_tre_mean_px']:.2f}px "
                   f"(n_landmarks={row['tre_n_landmarks']}) | {summary_str}")


def summarize(df, modes):
    print("\n" + "=" * 70)
    print("STEP 5 — ROMA_MODE sweep (ncc_warp, more negative = better)")
    print("=" * 70)

    ncc_cols = [f"{m}_ncc_warp" for m in modes]
    print("Per-mode: n succeeding / mean / median ncc_warp")
    means = {}
    for mode, col in zip(modes, ncc_cols):
        valid = df[col].dropna()
        means[mode] = valid.mean() if len(valid) else np.nan
        print(f"  {mode:<12} n={len(valid):>3} mean={valid.mean():.5f} "
             f"median={valid.median():.5f}" if len(valid) else f"  {mode:<12} n=0")

    ranked = sorted([m for m in modes if not np.isnan(means[m])], key=lambda m: means[m])
    print(f"\nRanking (best/most-negative first): {' > '.join(ranked)}")

    if 'ck_only' in modes:
        print("\n--- Each mode vs ck_only baseline (paired on the same pairs) ---")
        base_col = "ck_only_ncc_warp"
        for mode in modes:
            if mode == 'ck_only':
                continue
            col = f"{mode}_ncc_warp"
            sub = df.dropna(subset=[base_col, col])
            if len(sub) == 0:
                print(f"  {mode:<12} no pairs with both succeeding")
                continue
            delta = sub[col] - sub[base_col]  # negative = mode beats ck_only
            n_better = int((delta < -1e-6).sum())
            n_worse  = int((delta > 1e-6).sum())
            n_tied   = len(sub) - n_better - n_worse
            line = (f"  {mode:<12} n={len(sub):>3} better={n_better} worse={n_worse} "
                   f"tied={n_tied} mean_delta={delta.mean():.5f}")
            try:
                from scipy.stats import wilcoxon
                _, p = wilcoxon(sub[base_col], sub[col])
                line += f" p={p:.4f}"
            except ImportError:
                pass
            print(line)

    # Friedman test across all modes at once, restricted to pairs where every mode succeeded
    complete = df.dropna(subset=ncc_cols)
    print(f"\nPairs where ALL {len(modes)} modes succeeded: {len(complete)} / {len(df)}")
    if len(complete) >= 6:
        try:
            from scipy.stats import friedmanchisquare
            stat, p = friedmanchisquare(*[complete[c] for c in ncc_cols])
            print(f"Friedman test across all modes (H0: no difference): p={p:.4f}")
        except ImportError:
            pass
    else:
        print("(Need >= 6 complete pairs for a meaningful Friedman test — skipped.)")

    # ─────────────────────────────────────────────────────────────────────
    # SAME COMPARISON, BUT BY LANDMARK STABILITY INSTEAD OF NCC
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAME COMPARISON, BUT BY LANDMARK STABILITY INSTEAD OF NCC (px, lower = better)")
    print("Reuses the L0 affine's own inlier correspondences (matching VALIS's")
    print("documented practice), sampling the SAME deformation fields used above.")
    print("NOTE: because these points also fit the affine, this measures whether")
    print("each mode's warp PRESERVES good alignment at already-resolved points,")
    print("not 'how much did this mode improve over L0' — l0_only_tre_mean_px is")
    print("a sanity check (expected near-zero), not a baseline to compare against.")
    print("Mode-vs-mode comparisons below are still valid and meaningful.")
    print("=" * 70)

    tre_cols = [f"{m}_tre_mean_px" for m in modes]
    if "l0_only_tre_mean_px" in df.columns:
        l0_valid = df["l0_only_tre_mean_px"].dropna()
        print(f"L0-only baseline (affine, no L1 correction): n={len(l0_valid)} "
             f"mean={l0_valid.mean():.2f}px median={l0_valid.median():.2f}px")

    tre_means = {}
    print("\nPer-mode: n succeeding / mean / median TRE (px)")
    for mode in modes:
        col = f"{mode}_tre_mean_px"
        valid = df[col].dropna()
        tre_means[mode] = valid.mean() if len(valid) else np.nan
        if len(valid):
            print(f"  {mode:<12} n={len(valid):>3} mean={valid.mean():.3f}px "
                 f"median={df[f'{mode}_tre_median_px'].dropna().median():.3f}px")
        else:
            print(f"  {mode:<12} n=0")

    tre_ranked = sorted([m for m in modes if not np.isnan(tre_means[m])], key=lambda m: tre_means[m])
    print(f"\nTRE ranking (best/lowest-error first): {' > '.join(tre_ranked)}")

    ncc_verdicts = {}
    tre_verdicts = {}
    if "ck_only" in modes:
        print("\n--- Each mode vs ck_only, by TRE (paired on the same pairs) ---")
        base_col = "ck_only_tre_mean_px"
        for mode in modes:
            if mode == "ck_only":
                continue
            col = f"{mode}_tre_mean_px"
            sub = df.dropna(subset=[base_col, col])
            if len(sub) == 0:
                print(f"  {mode:<12} no pairs with both succeeding")
                continue
            delta = sub[col] - sub[base_col]  # negative = mode has LOWER (better) TRE than ck_only
            n_better = int((delta < -1e-6).sum())
            n_worse  = int((delta > 1e-6).sum())
            n_tied   = len(sub) - n_better - n_worse
            line = (f"  {mode:<12} n={len(sub):>3} better={n_better} worse={n_worse} "
                   f"tied={n_tied} mean_delta={delta.mean():+.3f}px")
            p_tre = np.nan
            try:
                from scipy.stats import wilcoxon
                _, p_tre = wilcoxon(sub[base_col], sub[col])
                line += f" p={p_tre:.4f}"
            except ImportError:
                pass
            print(line)
            tre_verdicts[mode] = (n_better > n_worse)

            # cross-check against the NCC verdict for this same mode
            ncc_sub = df.dropna(subset=["ck_only_ncc_warp", f"{mode}_ncc_warp"])
            if len(ncc_sub) > 0:
                ncc_delta = ncc_sub[f"{mode}_ncc_warp"] - ncc_sub["ck_only_ncc_warp"]
                ncc_verdicts[mode] = (int((ncc_delta < -1e-6).sum()) > int((ncc_delta > 1e-6).sum()))

    print("\n--- Agreement check: does TRE confirm the NCC-based verdict for each mode? ---")
    for mode in modes:
        if mode == "ck_only" or mode not in tre_verdicts or mode not in ncc_verdicts:
            continue
        agree = tre_verdicts[mode] == ncc_verdicts[mode]
        ncc_dir = "better" if ncc_verdicts[mode] else "worse"
        tre_dir = "better" if tre_verdicts[mode] else "worse"
        tag = "AGREE" if agree else "DISAGREE — do not trust the NCC verdict alone for this mode"
        print(f"  {mode:<12} NCC says {ncc_dir:<7} vs ck_only | TRE says {tre_dir:<7} vs ck_only  --> {tag}")

    if tre_ranked:
        print(f"\nTRE-best mode: '{tre_ranked[0]}'. Compare this to the NCC ranking above — "
             f"if they disagree on which mode wins outright, that disagreement is the more "
             f"important finding than either ranking on its own.")

    print("\n--- Coverage / confidence / timing per mode ---")
    for mode in modes:
        cov_col, conf_col, time_col = f"{mode}_coverage_pct", f"{mode}_mean_conf", f"{mode}_time_s"
        print(f"  {mode:<12} coverage={df[cov_col].mean():.1f}% "
             f"confidence={df[conf_col].mean():.3f} time={df[time_col].mean():.2f}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Isolated ROMA_MODE ablation.")
    parser.add_argument("--core_name", type=str, action="append", default=[])
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--modes", type=str, default=None,
                        help=f"Comma-separated subset to test. Default: all of "
                             f"{ALL_MODES}. E.g. --modes ck_only,dapi_clahe")
    args = parser.parse_args()

    core_names = list(args.core_name)
    if args.start is not None and args.end is not None:
        core_names += [f"Core_{i:02d}" for i in range(args.start, args.end + 1)]
    if not core_names:
        parser.error("Provide at least one --core_name, or both --start and --end.")

    modes = [m.strip() for m in args.modes.split(",")] if args.modes else list(ALL_MODES)
    for m in modes:
        if m not in ALL_MODES:
            parser.error(f"Unknown mode '{m}'. Choose from {ALL_MODES}.")

    total  = len(core_names)
    rows   = []
    status = {}

    logger.info("=" * 60)
    logger.info(f"ROMA_MODE ablation — {total} core(s), modes={modes}")
    logger.info("=" * 60)

    for idx, core_name in enumerate(core_names, start=1):
        logger.info(f"[{idx}/{total}] {core_name}")
        n_before = len(rows)
        try:
            process_core(core_name, rows, modes)
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

    summarize(df, modes)


if __name__ == "__main__":
    main()