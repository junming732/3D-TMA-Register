"""
Parametric ROMA_MODE Ablation Study
======================================================================
Evaluates the effect of varying normalization methods, channel weightings,
and tensor compilation strategies on RoMaV2 dense matching.

Outputs a "long format" CSV where each row represents a single configuration
execution for a specific slice pair. Computes both the pseudo-TRE (based on
L0 inliers) and the real-TRE (based on manual annotations) side-by-side.

Usage:
    python test_roma_mode_ablation.py --core_list 9,16,19
"""

import os
import sys
import glob
import re
import argparse
import logging
import time
import gc
import json
import itertools

import numpy as np
import pandas as pd
import tifffile
import cv2
import SimpleITK as sitk
import yaml
from skimage import exposure

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
import config
from held_out_tre import compute_tre_warp, compute_tre_affine

# Import the exact index parser from your evaluation logic
sys.path.append(os.path.join(parent_dir, "evaluation"))
try:
    from landmark_accuracy_common import z_json_to_slice_idx
except ImportError:
    logging.warning("Could not import landmark_accuracy_common. Real TRE will fail if landmarks are present.")
    def z_json_to_slice_idx(z): return int(z) - 1

os.environ['TORCH_HOME'] = os.path.join(config.DATASPACE, 'model_weights')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# Mask the GPU completely from this process before PyTorch initializes
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
torch._dynamo.config.disable = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
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
L0_METHOD                = cv2.RANSAC

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

DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")
OUTPUT_CSV        = os.path.join(current_dir, "tre_roma_mode_ablation.csv")

PIXEL_SIZE_UM = 0.4961


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRIC CONFIGURATION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_ablation_grid():
    """
    Generates a Cartesian product grid of parameters to test.
    """
    norm_methods = ['clahe', 'log1p']
    output_formats = ['direct_rgb', 'registration_color_lut', 'visual_additive_rgb']
    
    # Explicit names fix the confusing log output.
    # NOTE: '3ch_fusion' below is genuinely 3-channel (DAPI+CK+AF) only for
    # output_type='direct_rgb', which assigns raw channels straight to R/G/B.
    # For 'registration_color_lut' and 'visual_additive_rgb', AF (channel 7)
    # used to have no defined LUT/tint color, so it silently dropped out and
    # '3ch_fusion' was actually only DAPI+CK there. COLOR_LUT and
    # VISUAL_COLORS below now both define an AF entry, so '3ch_fusion' is
    # genuinely 3-channel everywhere. '2ch_fusion' is added explicitly so the
    # previous (AF-excluded) DAPI+CK behaviour is still directly testable and
    # clearly labeled, rather than being an accidental side effect.
    weight_combinations = [
        {'name': 'DAPI_only',  'weights': {0: 1.0}},
        {'name': 'CK_only',    'weights': {6: 1.0}},
        {'name': '2ch_fusion', 'weights': {0: 0.5,  6: 0.5}},
        {'name': '3ch_fusion', 'weights': {0: 0.33, 6: 0.33, 7: 0.33}},
        {'name': '7ch_fusion', 'weights': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}}
    ]
    
    grid = []
    for norm, fmt, combo in itertools.product(norm_methods, output_formats, weight_combinations):
        
        # 1. Prevent 7 channels from squeezing into a basic 3-slot RGB tensor
        if fmt == 'direct_rgb' and combo['name'] == '7ch_fusion':
            continue
            
        # 2. Prevent Single channels from wasting time in the multi-color blending scripts
        if fmt in ['registration_color_lut', 'visual_additive_rgb'] and len(combo['weights']) == 1:
            continue
            
        config_name = f"{norm}_{fmt}_{combo['name']}"
        grid.append({
            'mode_name': config_name,
            'norm_method': norm,
            'output_type': fmt,
            'weights': combo['weights']
        })
    return grid

def normalize_channel(img_arr: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """Normalizes a raw array to a uint8 array based on the specified method."""
    img_float = img_arr.astype(np.float32)
    if method == 'clahe':
        img01 = exposure.rescale_intensity(img_float, out_range=(0, 1))
        eq = exposure.equalize_adapthist(img01)
        return exposure.rescale_intensity(eq, out_range=(0, 255)).astype(np.uint8)
    elif method == 'log1p':
        log_img = np.log1p(img_float)
        p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
        return cv2.normalize(np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif method == 'linear':
        p_lo, p_hi = np.percentile(img_float[::4, ::4], (1.0, 99.0))
        return cv2.normalize(np.clip(img_float, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    raise ValueError(f"Unknown method: {method}")

def build_parametric_roma_input(vol: np.ndarray, weights: dict, norm_method: str, output_type: str) -> np.ndarray:
    """Builds the final tensor array for RoMaV2 processing."""
    h, w = vol.shape[1], vol.shape[2]
    total_weight = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
    
    if output_type == 'grayscale_duplicate':
        accumulator = np.zeros((h, w), dtype=np.float32)
        for ch_idx, weight in weights.items():
            if weight > 0:
                norm_img = normalize_channel(vol[ch_idx], method=norm_method).astype(np.float32)
                accumulator += norm_img * (weight / total_weight)
        gray_uint8 = np.clip(accumulator, 0, 255).astype(np.uint8)
        return np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)
        
    elif output_type == 'direct_rgb':
        sorted_channels = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:3]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i, (ch_idx, _) in enumerate(sorted_channels):
            if i < 3:
                rgb[..., i] = normalize_channel(vol[ch_idx], method=norm_method)
        return rgb

    elif output_type == 'registration_color_lut':
        # Channels 0-6 replicate akaze_romav2_multi_channel_warp_new.py's
        # COLOR_LUT / prepare_color_lut_fusion() exactly (values copied verbatim).
        # That function is explicitly "7-channel (AF excluded)" — its COLOR_LUT
        # never defines an entry for channel 7 at all, unlike convert_tiff_RGB_script.py
        # (which at least had an unused AF slot). So there's no reference value for
        # AF here; (160, 160, 160) below is an ablation-only invention with nothing
        # to inherit from. It happens not to collide in hue with any other channel
        # (all others are saturated colors; gray is desaturated), so unlike AF's
        # tint in visual_additive_rgb, this one doesn't need a fix — just flagging
        # that "replicates akaze_romav2_multi_channel_warp_new.py" only applies to
        # channels 0-6, not to AF's presence or color here.
        COLOR_LUT = {
            0: (0, 128, 255), 1: (51, 255, 51), 2: (255, 51, 51),
            3: (0, 255, 255), 4: (255, 0, 255), 5: (255, 255, 0),
            6: (255, 128, 0), 7: (160, 160, 160),  # AF — neutral gray, no hue of its own
        }
        acc = np.zeros((h, w, 3), dtype=np.float32)
        n_channels = 0
        for ch_idx, weight in weights.items():
            if weight > 0 and ch_idx in COLOR_LUT:
                norm_img = normalize_channel(vol[ch_idx], method=norm_method).astype(np.float32) / 255.0
                color_arr = np.array(COLOR_LUT[ch_idx], dtype=np.float32) / 255.0
                acc += norm_img[..., None] * color_arr[None, None, :]
                n_channels += 1
        # Averages by dividing by n_channels to prevent clipping
        return np.clip(acc / max(n_channels, 1) * 255.0, 0, 255).astype(np.uint8)

    elif output_type == 'visual_additive_rgb':
        # Replicates convert_tiff_RGB_script.py (Additive Fusion with Clipping)
        # for channels 0-6, copied verbatim from that script's `colors` array.
        # AF (channel 7) uses that same script's index-7 value, orange — present
        # in the array but never rendered there, since its loop is
        # `for channel in range(7)`. Previously this used white ([1,1,1]/4),
        # the same hue as DAPI (channel 0, also white) at lower intensity — AF
        # and DAPI were visually indistinguishable by color as a result. Orange
        # fixes that. NOTE: this changes the actual pixel values fed to RoMaV2
        # for every 3ch_fusion/7ch_fusion config under this output type (orange
        # has different total intensity than white), so numbers from any run
        # before this change are not directly comparable to a rerun.
        VISUAL_COLORS = {
            0: np.array([1, 1, 1]) / 2,
            1: np.array([0, 1, 0]) / 4,
            2: np.array([1, 1, 0]) / 4,
            3: np.array([1, 0, 1]) / 4,
            4: np.array([0, 1, 1]) / 4,
            5: np.array([1, 0, 0]) / 4,
            6: np.array([0.5, 0, 1]) / 4,
            7: np.array([1, 0.5, 0]) / 4,  # AF — orange, matches convert_tiff_RGB_script.py's unused index-7 entry
        }
        rgb_acc = np.zeros((h, w, 3), dtype=np.float32)
        for ch_idx, weight in weights.items():
            if weight > 0 and ch_idx in VISUAL_COLORS:
                norm_img = normalize_channel(vol[ch_idx], method=norm_method).astype(np.float32) / 255.0
                rgb_acc += norm_img[..., None] * VISUAL_COLORS[ch_idx]
        # Forcefully clips at 1.0, preserving visual brightness but crushing gradients
        rgb_acc = np.clip(rgb_acc, 0, 1)
        return (rgb_acc * 255).clip(0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown output type: {output_type}")

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

def prepare_ck_log(img_arr):
    img_float = img_arr.astype(np.float32)
    log_img = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    return cv2.normalize(np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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
    return None

def constrain_affine(M):
    if M is None: return None
    M_out = M.copy()
    U, S, Vt = np.linalg.svd(M_out[:2, :2])
    S = np.clip(S, 1.0 - MAX_SCALE_DEVIATION, 1.0 + MAX_SCALE_DEVIATION)
    if S[1] > 1e-6 and S[0] / S[1] > 1.0 + MAX_SHEAR:
        S[0] = S[1] * (1.0 + MAX_SHEAR)
    M_out[:2, :2] = U @ np.diag(S) @ Vt
    return M_out

def transform_is_sane(M):
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R = U @ Vt
    rot_deg = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return rot_deg <= MAX_ROTATION_DEG

def measure_ncc(fixed_f32, moving_f32, mask_uint8):
    try:
        sitk_f = sitk.GetImageFromArray(fixed_f32)
        sitk_m = sitk.GetImageFromArray(moving_f32)
        reg = sitk.ImageRegistrationMethod()
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
    kp1_raw, des1 = detector.detectAndCompute(fixed_log, fixed_mask)
    kp2_raw, des2 = detector.detectAndCompute(moving_log, moving_mask)
    if des1 is None or des2 is None or len(kp1_raw) < 4 or len(kp2_raw) < 4:
        return None, None, 0

    def cap_by_response(kps, des, max_kp):
        if len(kps) <= max_kp: return kps, des
        idx = np.argsort([kp.response for kp in kps])[::-1][:max_kp]
        return tuple(kps[i] for i in idx), des[idx]

    kp1, des1 = cap_by_response(kp1_raw, des1, AKAZE_MAX_KEYPOINTS)
    kp2, des2 = cap_by_response(kp2_raw, des2, AKAZE_MAX_KEYPOINTS)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        return None, None, len(good)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return src_pts, dst_pts, len(good)

def fit_affine(src_pts, dst_pts, slice_id):
    M, mask = cv2.estimateAffine2D(
        dst_pts, src_pts, method=L0_METHOD,
        ransacReprojThreshold=RANSAC_THRESH, maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE)
    if M is None or mask is None: return None, 0, None
    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS: return None, n_inliers, None
    M = constrain_affine(M)
    if M is None or not transform_is_sane(M): return None, n_inliers, None
    return M, n_inliers, mask

def to_rgb_pil(img):
    from PIL import Image
    if img.ndim == 2: return Image.fromarray(np.stack([img, img, img], axis=-1))
    elif img.ndim == 3 and img.shape[2] == 3: return Image.fromarray(img)
    raise ValueError(f"Unexpected image shape: {img.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# ROMAV2 LAZY LOAD
# ─────────────────────────────────────────────────────────────────────────────
_romav2_model = None

def get_romav2_model():
    global _romav2_model
    if _romav2_model is None:
        from romav2 import RoMaV2
        logger.info("Loading RoMaV2 on strict CPU...")
        _romav2_model = RoMaV2().to('cpu')
        _romav2_model.eval()
        _romav2_model = torch._dynamo.disable(_romav2_model)
        _romav2_model.H_lr = ROMAV2_H
        _romav2_model.W_lr = ROMAV2_W
        _romav2_model.H_hr = ROMAV2_H_HR
        _romav2_model.W_hr = ROMAV2_W_HR
    return _romav2_model

def call_romav2_match(img_A_uint8, img_B_uint8):
    model = get_romav2_model()
    img_A = to_rgb_pil(img_A_uint8)
    img_B = to_rgb_pil(img_B_uint8)
    try:
        with torch.no_grad():
            preds = model.match(img_A, img_B)
        warp_AB = preds['warp_AB'].squeeze(0).cpu().numpy().copy()
        overlap_AB = preds['overlap_AB'].squeeze().cpu().numpy().copy()
        overlap_AB = overlap_AB.reshape(warp_AB.shape[0], warp_AB.shape[1])
        return warp_AB, overlap_AB
    finally:
        gc.collect()

def _apply_cap_and_background(map_x, map_y, orig_h, orig_w, tissue_mask_full):
    identity_x_full, identity_y_full = np.meshgrid(
        np.arange(orig_w, dtype=np.float32), np.arange(orig_h, dtype=np.float32))
    disp_x = map_x - identity_x_full
    disp_y = map_y - identity_y_full
    mag = np.sqrt(disp_x**2 + disp_y**2)
    excess = mag > WARP_MAX_DISPLACEMENT_PX
    if np.any(excess):
        scale = np.where(excess, WARP_MAX_DISPLACEMENT_PX / (mag + 1e-8), 1.0)
        disp_x *= scale
        disp_y *= scale
        map_x = (identity_x_full + disp_x).astype(np.float32)
        map_y = (identity_y_full + disp_y).astype(np.float32)
    if tissue_mask_full is not None:
        background = ~(tissue_mask_full.astype(bool))
        if np.any(background):
            map_x = map_x.copy()
            map_y = map_y.copy()
            map_x[background] = identity_x_full[background]
            map_y[background] = identity_y_full[background]
    return map_x, map_y

def romav2_dense_warp_whole(fixed_input, moving_input, orig_h, orig_w, tissue_mask_full=None):
    try:
        warp_AB, overlap_AB = call_romav2_match(fixed_input, moving_input)
        H_lr, W_lr = warp_AB.shape[:2]

        b_coords_x = (warp_AB[..., 0] + 1.0) / 2.0 * (orig_w - 1)
        b_coords_y = (warp_AB[..., 1] + 1.0) / 2.0 * (orig_h - 1)

        confident_2d = overlap_AB >= WARP_CONFIDENCE_THRESH
        n_confident = int(confident_2d.sum())
        coverage_pct = n_confident / (H_lr * W_lr) * 100
        mean_confidence = float(overlap_AB.mean())

        grid_x_lr = np.linspace(0, orig_w - 1, W_lr, dtype=np.float32)
        grid_y_lr = np.linspace(0, orig_h - 1, H_lr, dtype=np.float32)
        identity_x, identity_y = np.meshgrid(grid_x_lr, grid_y_lr)

        map_x_lr = np.where(confident_2d, b_coords_x, identity_x).astype(np.float32)
        map_y_lr = np.where(confident_2d, b_coords_y, identity_y).astype(np.float32)

        map_x = cv2.resize(map_x_lr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(map_y_lr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        map_x, map_y = _apply_cap_and_background(map_x, map_y, orig_h, orig_w, tissue_mask_full)

        return map_x, map_y, coverage_pct, mean_confidence
    except Exception as exc:
        logger.warning(f"RoMaV2 failed: {exc}")
        return None, None, 0.0, 0.0

# ─────────────────────────────────────────────────────────────────────────────
# CORE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def load_full_volume(file_path):
    return tifffile.imread(file_path).astype(np.float32)

def process_core(core_name, rows):
    input_dir = os.path.join(DATA_BASE_PATH, core_name)
    if not os.path.exists(input_dir):
        logger.error(f"[{core_name}] Input folder not found: {input_dir}")
        return

    sample_files = sorted(
        glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff")),
        key=get_slice_number
    )
    file_list = [f for f in sample_files if "_thumb" not in os.path.basename(f)]

    allowed = load_slice_filter(SLICE_FILTER_YAML, core_name)
    if allowed is not None:
        file_list = [f for f in file_list if get_slice_number(f) in allowed]

    if len(file_list) < 2:
        logger.error(f"[{core_name}] Fewer than 2 slices after filtering.")
        return

    # Load Real Landmarks
    annotation_dir = os.path.join(parent_dir, "annotations")
    json_exact = os.path.join(annotation_dir, f"landmark_annotation_{core_name}.json")
    json_lower = os.path.join(annotation_dir, f"landmark_annotation_{core_name.lower()}.json")
    json_path = json_exact if os.path.exists(json_exact) else (json_lower if os.path.exists(json_lower) else None)

    real_ann_by_slice = {}
    unfiltered_file_list = [f for f in sample_files if "_thumb" not in os.path.basename(f)]
    file_to_orig_pos = {f: idx for idx, f in enumerate(unfiltered_file_list)}

    if json_path is not None:
        with open(json_path, 'r') as fh:
            ann_data = json.load(fh)
        for ann in ann_data:
            s_idx = z_json_to_slice_idx(ann['z'])
            mc = ann['landmark_id']
            if s_idx not in real_ann_by_slice:
                real_ann_by_slice[s_idx] = {}
            real_ann_by_slice[s_idx][mc] = (ann['x'], ann['y'])
        logger.info(f"[{core_name}] Loaded real annotations for {len(real_ann_by_slice)} slices.")
    else:
        logger.info(f"[{core_name}] No real landmark JSON found.")

    grid_configs = generate_ablation_grid()
    
    for i in range(len(file_list) - 1):
        file_a, file_b = file_list[i], file_list[i + 1]
        sid = f"{core_name}_Z{i:03d}-Z{i+1:03d}"

        vol_fixed = load_full_volume(file_a)
        vol_moving = load_full_volume(file_b)
        h, w = vol_fixed.shape[1], vol_fixed.shape[2]

        fixed_mask = load_mask_or_none(file_a, (h, w))
        moving_mask = load_mask_or_none(file_b, (h, w))

        tissue_frac = (np.count_nonzero(fixed_mask) / fixed_mask.size if fixed_mask is not None else 1.0)
        if tissue_frac < MASK_MIN_FRAC: continue

        # Extract Real Pairs
        real_fixed_pts, real_moving_pts = None, None
        if real_ann_by_slice:
            sidx_a = file_to_orig_pos.get(file_a)
            sidx_b = file_to_orig_pos.get(file_b)
            if sidx_a in real_ann_by_slice and sidx_b in real_ann_by_slice:
                shared_ids = set(real_ann_by_slice[sidx_a].keys()).intersection(real_ann_by_slice[sidx_b].keys())
                if shared_ids:
                    shared_ids = sorted(list(shared_ids))
                    real_fixed_pts = np.array([real_ann_by_slice[sidx_a][mc] for mc in shared_ids], dtype=np.float32)
                    real_moving_pts = np.array([real_ann_by_slice[sidx_b][mc] for mc in shared_ids], dtype=np.float32)

        # Baseline AKAZE (CK Log)
        fixed_log = prepare_ck_log(vol_fixed[CK_CHANNEL_IDX])
        moving_log = prepare_ck_log(vol_moving[CK_CHANNEL_IDX])
        src_pts, dst_pts, n_matches = detect_and_match(fixed_log, moving_log, fixed_mask, moving_mask, sid)
        
        if src_pts is None: continue
        M, n_inliers, inlier_mask = fit_affine(src_pts, dst_pts, sid)
        if M is None: continue

        moving_log_affine = cv2.warpAffine(
            moving_log, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ncc_affine = measure_ncc(fixed_log.astype(np.float32), moving_log_affine.astype(np.float32), fixed_mask)

        landmarks_fixed = src_pts[inlier_mask.ravel() == 1].reshape(-1, 2)
        landmarks_moving = dst_pts[inlier_mask.ravel() == 1].reshape(-1, 2)
        image_diag_px = float(np.hypot(h, w))
        ground_truth_targets = (M[:, :2] @ landmarks_moving.T).T + M[:, 2]

        real_moving_targets = (M[:, :2] @ real_moving_pts.T).T + M[:, 2] if real_moving_pts is not None else None
        
        l0_tre = compute_tre_affine(M, landmarks_fixed, landmarks_moving, px_size_um=PIXEL_SIZE_UM, image_diag_px=image_diag_px)
        
         # Parametric Grid Sweep
        for config in grid_configs:
            mode_name = config['mode_name']
            
            fixed_input = build_parametric_roma_input(vol_fixed, config['weights'], config['norm_method'], config['output_type'])
            moving_input = build_parametric_roma_input(vol_moving, config['weights'], config['norm_method'], config['output_type'])
            
            moving_input_affine = cv2.warpAffine(
                moving_input, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            t0 = time.time()
            map_x, map_y, cov, meanconf = romav2_dense_warp_whole(
                fixed_input, moving_input_affine, h, w, tissue_mask_full=fixed_mask)
            elapsed = time.time() - t0

            row_data = {
                'core': core_name,
                'pair': sid,
                'mode_name': mode_name,
                'norm_method': config['norm_method'],
                'output_type': config['output_type'],
                'w_dapi': config['weights'].get(0, 0.0),
                'w_ck': config['weights'].get(6, 0.0),
                'w_af': config['weights'].get(7, 0.0),
                'n_matches': n_matches,
                'n_inliers': n_inliers,
                'ncc_affine': ncc_affine,
                'l0_only_tre_mean_px': l0_tre["mean_px"],
                'time_s': elapsed,
                'coverage_pct': cov,
                'mean_conf': meanconf
            }

            if map_x is not None:
                warped_ck = cv2.remap(moving_log_affine.astype(np.float32), map_x, map_y,
                                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                row_data['ncc_warp'] = measure_ncc(fixed_log.astype(np.float32), warped_ck, fixed_mask)

                tre = compute_tre_warp(map_x, map_y, landmarks_fixed, ground_truth_targets,
                                       px_size_um=PIXEL_SIZE_UM, image_diag_px=image_diag_px)
                row_data['pseudo_tre_mean_px'] = tre["mean_px"]
                row_data['pseudo_tre_median_px'] = tre["median_px"]
                
                # Real TRE
                if real_fixed_pts is not None and len(real_fixed_pts) > 0:
                    
                    real_tre = compute_tre_warp(map_x, map_y, real_fixed_pts, real_moving_targets,
                                                px_size_um=PIXEL_SIZE_UM, image_diag_px=image_diag_px)
                    row_data['real_tre_mean_px'] = real_tre["mean_px"]
                    row_data['real_tre_median_px'] = real_tre["median_px"]
                else:
                    row_data['real_tre_mean_px'] = np.nan
                    row_data['real_tre_median_px'] = np.nan
            else:
                row_data['ncc_warp'] = np.nan
                row_data['pseudo_tre_mean_px'] = np.nan
                row_data['pseudo_tre_median_px'] = np.nan
                row_data['real_tre_mean_px'] = np.nan
                row_data['real_tre_median_px'] = np.nan

            rows.append(row_data)
            logger.info(
                f"[{sid}] {mode_name}: "
                f"ncc={row_data.get('ncc_warp', float('nan')):.4f} | "
                f"pseudo_tre={row_data.get('pseudo_tre_mean_px', float('nan')):.2f}px | "
                f"real_tre={row_data.get('real_tre_mean_px', float('nan')):.2f}px"
            )
# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZE
# ─────────────────────────────────────────────────────────────────────────────
def summarize(df):
    print("\n" + "=" * 100)
    print("PARAMETRIC GRID ABLATION SUMMARY")
    print("=" * 100)
    
    # Aggregate stats per mode_name
    summary_df = df.groupby('mode_name').agg({
        'ncc_warp': ['count', 'mean'],
        'pseudo_tre_mean_px': 'mean',
        'real_tre_mean_px': 'mean',
        'coverage_pct': 'mean',
        'time_s': 'mean'
    }).reset_index()
    
    # Flatten multi-level columns
    summary_df.columns = ['Mode_Name', 'Success_Count', 'NCC_Warp_Mean', 
                          'Pseudo_TRE_Mean', 'Real_TRE_Mean', 'Coverage_Pct', 'Time_s']
    
    # Sort by Real_TRE_Mean if available, otherwise Pseudo_TRE_Mean
    if not summary_df['Real_TRE_Mean'].isna().all():
        summary_df = summary_df.sort_values(by='Real_TRE_Mean', ascending=True)
        print("Sorted by Real TRE (lower is better).")
    else:
        summary_df = summary_df.sort_values(by='Pseudo_TRE_Mean', ascending=True)
        print("Sorted by Pseudo TRE (lower is better).")
        
    print("\n")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 100)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Parametric ROMA_MODE Ablation.")
    parser.add_argument("--core_name", type=str, action="append", default=[])
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--core_list", type=str, default=None,
                        help="Comma-separated list of core numbers (e.g., 9,16,19)")
    args = parser.parse_args()

    core_names = list(args.core_name)
    if args.start is not None and args.end is not None:
        core_names += [f"Core_{i:02d}" for i in range(args.start, args.end + 1)]
        
    if args.core_list:
        for c in args.core_list.split(','):
            c = c.strip()
            if c.isdigit():
                core_names.append(f"Core_{int(c):02d}")
            else:
                core_names.append(c)

    if not core_names:
        parser.error("Provide at least one --core_name, --core_list, or both --start and --end.")

    total = len(core_names)
    rows = []

    logger.info("=" * 60)
    logger.info(f"Parametric Ablation — {total} core(s)")
    logger.info("=" * 60)

    for idx, core_name in enumerate(core_names, start=1):
        logger.info(f"[{idx}/{total}] Processing {core_name}")
        try:
            process_core(core_name, rows)
        except Exception as exc:
            logger.error(f"[{core_name}] Crashed: {exc}")

    if not rows:
        logger.error("No valid pairs processed across any core.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Wrote {len(df)} configuration executions → {OUTPUT_CSV}")

    summarize(df)

if __name__ == "__main__":
    main()