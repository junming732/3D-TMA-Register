"""
Data-Driven Explainer: AKAZE Affine & RoMaV2 Non-linear Registration.
Generates 5 presentation-ready figures using real TMA core data.

Changes from v1:
  fig1 → 3-panel preprocessing comparison: raw | log-normalised | tissue-masked
  fig2 → keypoints are now filtered to tissue mask before plotting (no out-of-tissue points)
  fig3 → removed schematic "resize" and "upsample" panels; kept only real-data panels
  fig5 → Added grid visualization showing all multiplexed channels from the fixed image with pseudo-coloring.
  Pipeline → matches akaze_linear_romav2_warp.py exactly:
             AKAZE detection on log images with tissue mask
             RoMaV2 runs on linear images
             Tissue mask applied at LR resolution BEFORE upsampling (not after)
"""

import os
import sys
import argparse
import glob
import re
import numpy as np
import cv2
import tifffile
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

# Torch setup
os.environ['TORCH_HOME'] = os.path.join(config.DATASPACE, 'model_weights')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch._dynamo.config.disable = True

# ─────────────────────────────────────────────────────────────────────────────
# STYLE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
FONT_SANS   = 'DejaVu Sans'
C_FIXED     = '#00AEEF'   # Cyan
C_MOVING    = '#EC008C'   # Magenta
C_AFFINE    = '#4CAF50'
C_WARP      = '#9C27B0'
C_ARROW     = '#444444'

DPI = 200
plt.rcParams.update({
    'font.family':       FONT_SANS,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.spines.left':  False,
    'axes.spines.bottom':False,
    'axes.facecolor':    'white',
    'figure.facecolor':  'white',
    'text.color':        '#222222',
    'axes.labelcolor':   '#222222',
})

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CONFIGURATION  (must match akaze_linear_romav2_warp.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
AKAZE_THRESHOLD     = 0.0001
AKAZE_MAX_KEYPOINTS = 20_000
LOWE_RATIO          = 0.80
MIN_MATCHES         = 20
MIN_INLIERS         = 6
RANSAC_CONFIDENCE   = 0.995
RANSAC_MAX_ITERS    = 5000
RANSAC_THRESH       = 8.0
MAX_SCALE_DEVIATION = 0.08
MAX_SHEAR           = 0.15
MAX_ROTATION_DEG    = 15.0

ROMAV2_DEVICE            = 'cpu'
ROMAV2_H, ROMAV2_W       = 448, 448
ROMAV2_H_HR              = None
ROMAV2_W_HR              = None
WARP_CONFIDENCE_THRESH   = 0.0
WARP_MAX_DISPLACEMENT_PX = 200.0
MASK_DILATE_PX           = 20
CK_CHANNEL_IDX           = 6

CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC & PREPROCESSING  (identical to akaze_linear_romav2_warp.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0

def conform_slice(arr, target_h, target_w):
    c, h, w = arr.shape
    out = np.zeros((c, target_h, target_w), dtype=arr.dtype)
    src_y0, dst_y0 = max(0, (h - target_h) // 2), max(0, (target_h - h) // 2)
    copy_h = min(h - src_y0, target_h - dst_y0)
    src_x0, dst_x0 = max(0, (w - target_w) // 2), max(0, (target_w - w) // 2)
    copy_w = min(w - src_x0, target_w - dst_x0)
    out[:, dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = arr[:, src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
    return out

def prepare_ck(img_arr):
    """
    Returns (norm_lin, norm_log) — both uint8.
    norm_lin : linear percentile stretch → fed to RoMaV2
    norm_log : log-stretch then percentile normalise → used for AKAZE, tissue masking, NCC
    Matches akaze_linear_romav2_warp.py prepare_ck exactly.
    """
    img_float = img_arr.astype(np.float32)

    # Linear normalisation — for RoMaV2
    p_lo_lin, p_hi_lin = np.percentile(img_float[::4, ::4], (0.1, 99.9))
    norm_lin = cv2.normalize(
        np.clip(img_float, p_lo_lin, p_hi_lin), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Log normalisation — for AKAZE, tissue mask, NCC
    log_img    = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log   = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return norm_lin, norm_log

def build_tissue_mask(ck_log: np.ndarray) -> np.ndarray:
    """
    Generates a tissue mask matching the exact dual-path segmentation logic 
    from the cropping pipeline (Safe Mask + Triangle Thresholding).
    
    Args:
        ck_log (np.ndarray): Input uint8 array (log-normalized CK channel).
        
    Returns:
        np.ndarray: Binary mask (0 and 255) of the same shape.
    """
    img = ck_log.astype(np.uint8)
    
    # 1. Generate "Safe Mask" (Background Removal)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
    bg_est = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_bg)
    foreground_rough = cv2.subtract(img, bg_est)
    
    _, rough_mask = cv2.threshold(foreground_rough, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    safe_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_DILATE, kernel_safe)
    
    # 2. Contrast Enhancement (Linear Stretch equivalent inline)
    nonzero = img[img > 0]
    if len(nonzero) > 0:
        p_min, p_max = np.percentile(nonzero, (0.5, 99.5))
        stretched_img = np.clip((img - p_min) / (p_max - p_min + 1e-5) * 255, 0, 255).astype(np.uint8)
    else:
        stretched_img = img.copy()

    # 3. Segmentation (Triangle Method on blurred image)
    blur = cv2.GaussianBlur(stretched_img, (15, 15), 0)
    _, binary_raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    
    # 4. Apply Safe Mask (Intersection)
    binary_masked = cv2.bitwise_and(binary_raw, binary_raw, mask=safe_mask)
    
    # 5. Morphology Cleanup
    CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    closed = cv2.morphologyEx(binary_masked, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=2)
    final_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=2)
    
    return final_mask

def measure_ncc(fixed_f32, moving_f32, mask_uint8):
    try:
        sitk_f, sitk_m = sitk.GetImageFromArray(fixed_f32), sitk.GetImageFromArray(moving_f32)
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
    except:
        return 0.0

def constrain_affine(M: np.ndarray) -> np.ndarray:
    if M is None: return None
    M_out = M.copy()
    U, S, Vt = np.linalg.svd(M_out[:2, :2])
    S = np.clip(S, 1.0 - MAX_SCALE_DEVIATION, 1.0 + MAX_SCALE_DEVIATION)
    if S[1] > 1e-6 and S[0] / S[1] > 1.0 + MAX_SHEAR:
        S[0] = S[1] * (1.0 + MAX_SHEAR)
    M_out[:2, :2] = U @ np.diag(S) @ Vt
    return M_out

def transform_is_sane(M: np.ndarray) -> bool:
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R = U @ Vt
    rot_deg = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return rot_deg <= MAX_ROTATION_DEG

def akaze_affine(fixed_log, moving_log, fixed_mask, moving_mask):
    """
    Tissue-masked AKAZE detection → BFMatcher → RANSAC affine.
    Matches akaze_linear_romav2_warp.py exactly.
    Returns (M, kp1, kp2, good, inlier_mask).
    """
    detector = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    kp1_raw, des1 = detector.detectAndCompute(fixed_log, fixed_mask)
    kp2_raw, des2 = detector.detectAndCompute(moving_log, moving_mask)

    if des1 is None or des2 is None:
        return None, [], [], [], []

    # Cap by response
    def cap(kps, des, max_kp):
        if len(kps) <= max_kp: return kps, des
        idx = np.argsort([kp.response for kp in kps])[::-1][:max_kp]
        return tuple(kps[i] for i in idx), des[idx]

    kp1, des1 = cap(kp1_raw, des1, AKAZE_MAX_KEYPOINTS)
    kp2, des2 = cap(kp2_raw, des2, AKAZE_MAX_KEYPOINTS)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        return None, kp1, kp2, good, []

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inlier_mask = cv2.estimateAffine2D(
        dst_pts, src_pts, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
    )

    M = constrain_affine(M)
    if M is None or not transform_is_sane(M):
        return None, kp1, kp2, good, inlier_mask
    return M, kp1, kp2, good, inlier_mask

def ck_to_rgb_pil(ck_uint8):
    from PIL import Image
    return Image.fromarray(np.stack([ck_uint8, ck_uint8, ck_uint8], axis=-1))

def romav2_dense_warp(fixed_lin, moving_lin_affine, orig_h, orig_w, tissue_mask_full):
    """
    RoMaV2 dense warp. Matches akaze_linear_romav2_warp.py romav2_dense_warp exactly:
    - Runs on linear images (not log)
    - Tissue mask applied at LR resolution BEFORE upsampling
    - Background cells forced to identity at LR to prevent seam artefacts
    """
    from romav2 import RoMaV2
    model = RoMaV2().to(ROMAV2_DEVICE)
    model.eval()
    model = torch._dynamo.disable(model)
    model.H_lr = ROMAV2_H
    model.W_lr = ROMAV2_W
    model.H_hr = ROMAV2_H_HR
    model.W_hr = ROMAV2_W_HR

    img_A = ck_to_rgb_pil(fixed_lin)
    img_B = ck_to_rgb_pil(moving_lin_affine)

    with torch.no_grad():
        preds = model.match(img_A, img_B)

    warp_AB    = preds['warp_AB'].squeeze(0).cpu().numpy()
    overlap_AB = preds['overlap_AB'].squeeze().cpu().numpy()
    H_lr, W_lr = warp_AB.shape[:2]

    b_coords_x = (warp_AB[..., 0] + 1.0) / 2.0 * (orig_w - 1)
    b_coords_y = (warp_AB[..., 1] + 1.0) / 2.0 * (orig_h - 1)

    confident_2d = overlap_AB.reshape(H_lr, W_lr) >= WARP_CONFIDENCE_THRESH
    conf_full    = cv2.resize(overlap_AB.reshape(H_lr, W_lr), (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)

    grid_x_lr = np.linspace(0, orig_w - 1, W_lr, dtype=np.float32)
    grid_y_lr = np.linspace(0, orig_h - 1, H_lr, dtype=np.float32)
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

    # ── KEY FIX: apply tissue mask at LR resolution BEFORE upsampling ────────
    # (matches akaze_linear_romav2_warp.py exactly — prevents seam artefacts)
    if tissue_mask_full is not None:
        mask_lr    = cv2.resize(tissue_mask_full, (W_lr, H_lr),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        background = ~mask_lr
        if np.any(background):
            map_x_lr[background] = identity_x[background]
            map_y_lr[background] = identity_y[background]

    # Upsample to full resolution
    map_x = cv2.resize(map_x_lr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_lr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return map_x, map_y, conf_full

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION UTILS
# ─────────────────────────────────────────────────────────────────────────────

def norm01(arr, mask):
    p = np.percentile(arr[mask > 0], 99.5) if np.any(mask > 0) else 1.0
    result = np.clip(arr / p, 0, 1).astype(np.float32)
    result[mask == 0] = 0.0   # zero outside tissue for clean background
    return result

def create_cyan_magenta_overlay(f_float, m_float, mask_f, mask_m):
    """Fixed = Cyan. Moving = Magenta. Background = White."""
    f  = np.clip(f_float * 1.5, 0, 1)
    m  = np.clip(m_float * 1.5, 0, 1)
    R, G, B = 1.0 - f, 1.0 - m, np.ones_like(f)
    bg = (mask_f == 0) & (mask_m == 0)
    R[bg], G[bg], B[bg] = 1.0, 1.0, 1.0
    return np.dstack((R, G, B))

def create_single_tint(img_float, mask, color_type='cyan'):
    img = np.clip(img_float * 1.5, 0, 1)
    R, G, B = np.ones_like(img), np.ones_like(img), np.ones_like(img)
    if color_type == 'cyan':    R = 1.0 - img
    if color_type == 'magenta': G = 1.0 - img
    bg = mask == 0
    R[bg], G[bg], B[bg] = 1.0, 1.0, 1.0
    return np.dstack((R, G, B))

def create_faded_bg(img_float, mask):
    img = np.clip(img_float, 0, 1)
    faded = 1.0 - (img * 0.3)
    faded[mask == 0] = 1.0
    return np.dstack((faded, faded, faded))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Preprocessing comparison (raw | log-normalised | tissue-masked)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_preprocessing_comparison(out_path, f_ck_raw, f_lin, f_log, mask_f):
    """
    4-panel comparison of the fixed CK channel at each preprocessing stage:
      (a) Raw fluorescence (uint16 percentile-clipped)
      (b) Log-normalised   → used by AKAZE, tissue masking, NCC
      (c) Linear-normalised → fed directly to RoMaV2 (no log transform)
      (d) Tissue mask boundary overlaid on log image (contour only — pixels never modified)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=DPI)
    fig.suptitle('CK Channel — Preprocessing Stages (Fixed Slice)',
                 fontsize=14, fontweight='bold', y=1.02)

    # (a) Raw
    raw_float = f_ck_raw.astype(np.float32)
    p_hi_raw  = np.percentile(raw_float, 99.9)
    raw_disp  = np.clip(raw_float / (p_hi_raw if p_hi_raw > 0 else 1.0), 0.0, 1.0)
    axes[0].imshow(raw_disp, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('(a) Raw fluorescence\n(uint16, percentile-clipped for display)',
                      fontsize=10, pad=10)

    # (b) Log-normalised — input to AKAZE, tissue mask, NCC
    log_disp = f_log.astype(np.float32) / 255.0
    axes[1].imshow(log_disp, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('(b) Log-normalised\n(→ AKAZE · tissue mask · NCC)',
                      fontsize=10, pad=10)

    # (c) Linear-normalised — what RoMaV2 actually receives
    lin_disp = f_lin.astype(np.float32) / 255.0
    axes[2].imshow(lin_disp, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('(c) Linear-normalised\n(→ RoMaV2 dense matching input)',
                      fontsize=10, pad=10)

    # (d) Tissue mask boundary — contour only, pixels never modified in pipeline
    axes[3].imshow(log_disp, cmap='gray', vmin=0, vmax=1)
    contours, _ = cv2.findContours(mask_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        pts = cnt[:, 0, :]
        axes[3].plot(pts[:, 0], pts[:, 1], color='#00AEEF', lw=1.2, alpha=0.9)
    axes[3].set_title('(d) Tissue mask boundary\n(cyan contour — pixels unchanged in pipeline)',
                      fontsize=10, pad=10)

    for ax in axes:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — AKAZE + RANSAC Affine stages
# ─────────────────────────────────────────────────────────────────────────────

def fig2_affine_stages(out_path, w, h, f_norm, m_raw_norm,
                        m_aff_norm, f_lin_norm, m_lin_aff_norm,
                        mask_f, mask_m, mask_m_aff,
                        kp1, kp2, good, inlier_mask):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.0), dpi=DPI)
    fig.suptitle('Stage 1 — AKAZE + RANSAC Affine Alignment',
                 fontsize=14, fontweight='bold', y=1.01)

    for ax in axes:
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')

    # (a) Raw log overlay — input to AKAZE
    axes[0].imshow(create_cyan_magenta_overlay(f_norm, m_raw_norm, mask_f, mask_m),
                   extent=[0, w, h, 0])
    axes[0].set_title("(a) Raw pair (log-normalised)\nCyan=Fixed, Magenta=Moving",
                      fontsize=10, pad=10)

    # (b) AKAZE keypoints on log image (AKAZE runs on log)
    axes[1].imshow(create_single_tint(f_norm, mask_f, 'cyan'),
                   extent=[0, w, h, 0], alpha=0.5)
    if kp1:
        pts = np.array([kp.pt for kp in kp1])
        xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
        in_tissue = mask_f[ys, xs] > 0
        pts_tissue = pts[in_tissue]
        if len(pts_tissue) > 0:
            axes[1].plot(pts_tissue[:, 0], pts_tissue[:, 1],
                         '.', ms=3, color='#004488', alpha=0.6)
    axes[1].set_title("(b) AKAZE keypoints\n(detected on log image, tissue-masked)",
                      fontsize=10, pad=10)

    # (c) RANSAC inlier matches
    axes[2].set_xlim(0, w*2.2)
    axes[2].imshow(create_single_tint(f_norm, mask_f, 'cyan'),
                   extent=[0, w, h, 0], alpha=0.3)
    axes[2].imshow(create_single_tint(m_raw_norm, mask_m, 'magenta'),
                   extent=[w*1.2, w*2.2, h, 0], alpha=0.3)
    if inlier_mask is not None and len(inlier_mask) > 0:
        inliers = [m for i, m in enumerate(good) if inlier_mask.ravel()[i]]
        if len(inliers) > 150:
            inliers = list(np.random.choice(inliers, 150, replace=False))
        for m in inliers:
            p1 = kp1[m.queryIdx].pt
            p2 = kp2[m.trainIdx].pt
            axes[2].plot([p1[0], p2[0] + w*1.2], [p1[1], p2[1]],
                         color=C_AFFINE, lw=0.4, alpha=0.7)
    axes[2].set_title("(c) RANSAC inlier\nmatches", fontsize=10, pad=10)

    # (d) After affine — log overlay (log used for NCC and visual QC)
    axes[3].imshow(create_cyan_magenta_overlay(f_norm, m_aff_norm, mask_f, mask_m_aff),
                   extent=[0, w, h, 0])
    axes[3].set_title("(d) After affine (log-normalised)\nDeep Blue = Perfect Overlap",
                      fontsize=10, pad=10)

    # (e) Affine-aligned linear pair — what is handed off to RoMaV2
    axes[4].imshow(create_cyan_magenta_overlay(f_lin_norm, m_lin_aff_norm, mask_f, mask_m_aff),
                   extent=[0, w, h, 0])
    axes[4].set_title("(e) Handoff to RoMaV2 (linear-normalised)\nwhat Stage 2 actually receives",
                      fontsize=10, pad=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — RoMaV2 dense warp (real-data panels only, no schematic boxes)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_romav2_grid(out_path, w, h, f_lin_norm, m_lin_aff_norm, f_norm, m_warp_norm,
                      mask_f, mask_m_aff, mask_m_warp, map_x, map_y, conf_full):
    """
    4-panel figure — what RoMaV2 actually receives and produces:
      (a) Input — linear-normalised affine-aligned pair (exactly what RoMaV2 sees)
      (b) Confidence map — overlap_AB tensor RoMaV2 returns
      (c) Displacement magnitude — derived from warp_AB tensor
      (d) Output — log overlay after applying the dense warp (for visual QC)
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), dpi=DPI)
    fig.suptitle('Stage 2 — RoMaV2 Dense Warp: Inputs & Outputs',
                 fontsize=14, fontweight='bold', y=1.02)

    for ax in axes:
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')

    bg_faded = np.ones((h, w, 3), dtype=np.float32)

    # (a) Linear input — exactly what RoMaV2 receives
    axes[0].imshow(create_cyan_magenta_overlay(f_lin_norm, m_lin_aff_norm, mask_f, mask_m_aff),
                   extent=[0, w, h, 0])
    axes[0].set_title('(a) RoMaV2 input (linear-normalised)\nwhat the model actually sees', fontsize=10, pad=8)

    # (b) Confidence — overlap_AB, one of two tensors RoMaV2 returns
    axes[1].imshow(bg_faded, extent=[0, w, h, 0])
    conf_masked = np.where(mask_f > 0, conf_full, np.nan)
    im_conf = axes[1].imshow(conf_masked, extent=[0, w, h, 0],
                              cmap='RdYlGn', vmin=0, vmax=1, alpha=0.85)
    plt.colorbar(im_conf, ax=axes[1], fraction=0.046, pad=0.04, label='overlap score')
    axes[1].set_title('(b) Match confidence (overlap_AB)\ngreen = certain, red = uncertain',
                      fontsize=10, pad=8)

    # (c) Displacement magnitude — derived from warp_AB, the other RoMaV2 tensor
    axes[2].imshow(bg_faded, extent=[0, w, h, 0])
    id_x_full  = np.arange(w, dtype=np.float32)[None, :]
    id_y_full  = np.arange(h, dtype=np.float32)[:, None]
    mag_full   = np.sqrt((map_x - id_x_full)**2 + (map_y - id_y_full)**2)
    mag_masked = np.where(mask_f > 0, mag_full, np.nan)
    vmax_val   = np.nanpercentile(mag_masked, 99) if np.any(mask_f > 0) else 1.0
    im_mag = axes[2].imshow(mag_masked, extent=[0, w, h, 0],
                             cmap='YlOrRd', vmin=0, vmax=vmax_val, alpha=0.85)
    plt.colorbar(im_mag, ax=axes[2], fraction=0.046, pad=0.04, label='pixels displaced')
    axes[2].set_title('(c) Displacement magnitude (from warp_AB)\npx of residual correction per location',
                      fontsize=10, pad=8)

    # (d) Output
    axes[3].imshow(create_cyan_magenta_overlay(f_norm, m_warp_norm, mask_f, mask_m_warp),
                   extent=[0, w, h, 0])
    axes[3].set_title('(d) RoMaV2 output\n(affine + dense warp applied)', fontsize=10, pad=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Warp field: displacement vectors + before/after result
# ─────────────────────────────────────────────────────────────────────────────

def fig4_warp_field_anatomy(out_path, w, h,
                             f_lin_norm, m_lin_norm, m_lin_aff_norm, m_lin_warp_norm,
                             mask_f, mask_m, mask_m_aff, mask_m_lin_warp, map_x, map_y):
    """
    3-panel direct comparison showing the full registration progression.
    All panels use LINEAR normalisation for a consistent visual space.

      (a) Raw Input:     Unaligned linear pair
      (b) RoMaV2 input:  Linear affine-aligned pair
      (c) After warp:    Linear warped pair
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)
    fig.suptitle('Registration Progression — Raw vs Affine vs Dense Warp\n'
                 '(all panels: linear-normalised, same image space)',
                 fontsize=13, fontweight='bold', y=1.03)
 
    for ax in axes:
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
 
    # (a) Raw Input — Unaligned
    axes[0].imshow(
        create_cyan_magenta_overlay(f_lin_norm, m_lin_norm, mask_f, mask_m),
        extent=[0, w, h, 0]
    )
    axes[0].set_title('(a) Raw Input (Unaligned)\n'
                      'Cyan=Fixed, Magenta=Moving',
                      fontsize=11, pad=10)

    # (b) Input to RoMaV2 — Linear affine-aligned
    axes[1].imshow(
        create_cyan_magenta_overlay(f_lin_norm, m_lin_aff_norm, mask_f, mask_m_aff),
        extent=[0, w, h, 0]
    )
    axes[1].set_title('(b) After Affine Alignment (RoMaV2 Input)\n'
                      'Cyan=Fixed, Magenta=Moving',
                      fontsize=11, pad=10)
 
    # (c) Output of RoMaV2 — Linear warped
    axes[2].imshow(
        create_cyan_magenta_overlay(f_lin_norm, m_lin_warp_norm, mask_f, mask_m_lin_warp),
        extent=[0, w, h, 0]
    )
    axes[2].set_title('(c) After Dense Warp (RoMaV2 Output)\n'
                      'Cyan=Fixed, Magenta=Moving',
                      fontsize=11, pad=10)
 
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — All Channels Fixed Image Grid
# ─────────────────────────────────────────────────────────────────────────────

def fig5_fixed_all_channels(out_path, f_arr, channel_names):
    """
    Generates a grid showing all multiplexed channels from the fixed image.
    Applies custom pseudo-coloring for clear visual distinction across channels.
    """
    num_channels = f_arr.shape[0]
    cols = min(4, num_channels)
    rows = int(np.ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=DPI)
    fig.suptitle('Multiplexed Channels — Fixed Image Reference', fontsize=16, fontweight='bold', y=1.02)

    if num_channels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Distinct pseudo-colors to clearly differentiate markers
    colors = [
        (0.0, 0.5, 1.0),   # Ch 0: Light Blue
        (0.2, 1.0, 0.2),   # Ch 1: Bright Green
        (1.0, 0.2, 0.2),   # Ch 2: Bright Red
        (0.0, 1.0, 1.0),   # Ch 3: Cyan
        (1.0, 0.0, 1.0),   # Ch 4: Magenta
        (1.0, 1.0, 0.0),   # Ch 5: Yellow
        (1.0, 0.5, 0.0),   # Ch 6: Orange
        (0.7, 0.7, 0.7),   # Ch 7: Gray
    ]

    for i in range(len(axes)):
        if i < num_channels:
            ch_data = f_arr[i].astype(np.float32)
            p_hi = np.percentile(ch_data, 99.9)
            if p_hi > 0:
                ch_norm = np.clip(ch_data / p_hi, 0, 1)
            else:
                ch_norm = ch_data

            rgb = np.zeros((*ch_norm.shape, 3), dtype=np.float32)
            color = colors[i % len(colors)]
            
            # Apply tint to grayscale array
            rgb[..., 0] = ch_norm * color[0]
            rgb[..., 1] = ch_norm * color[1]
            rgb[..., 2] = ch_norm * color[2]

            axes[i].imshow(rgb)
            name = channel_names[i] if i < len(channel_names) else f"Channel {i}"
            axes[i].set_title(f"[{i}] {name}", fontsize=12, fontweight='bold')
        
        # Hide axes for all panels (including empty ones if the grid is larger than num_channels)
        axes[i].axis('off')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate explainer figures with real data (Cyan/Magenta overlays).'
    )
    parser.add_argument('--core_name',  type=str, required=True)
    parser.add_argument('--fixed_id',   type=int, required=True)
    parser.add_argument('--moving_id',  type=int, required=True)
    args = parser.parse_args()

    IN_DIR  = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed", args.core_name)
    OUT_DIR = os.path.join(config.DATASPACE, "Final_Explainer_Plots", args.core_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    files  = glob.glob(os.path.join(IN_DIR, "*.ome.tif"))
    f_file = next((f for f in files if get_slice_number(f) == args.fixed_id),  None)
    m_file = next((f for f in files if get_slice_number(f) == args.moving_id), None)
    if not f_file or not m_file:
        sys.exit("Error: Files not found.")

    print(f"Loading data for Fixed ID {args.fixed_id} & Moving ID {args.moving_id}...")
    f_arr = tifffile.imread(f_file)
    m_arr = tifffile.imread(m_file)
    if f_arr.ndim == 3 and f_arr.shape[-1] < f_arr.shape[0]: f_arr = np.moveaxis(f_arr, -1, 0)
    if m_arr.ndim == 3 and m_arr.shape[-1] < m_arr.shape[0]: m_arr = np.moveaxis(m_arr, -1, 0)

    th, tw = f_arr.shape[1], f_arr.shape[2]
    if m_arr.shape[1] != th or m_arr.shape[2] != tw:
        m_arr = conform_slice(m_arr, th, tw)

    # ── Preprocessing (matches pipeline exactly) ─────────────────────────────
    f_ck_raw = f_arr[CK_CHANNEL_IDX]                       # uint16 raw — for fig1 only
    f_ck  = f_arr[CK_CHANNEL_IDX].astype(np.float32)
    m_ck  = m_arr[CK_CHANNEL_IDX].astype(np.float32)

    f_lin, f_log = prepare_ck(f_ck)                        # lin → RoMaV2, log → AKAZE/NCC/mask
    m_lin, m_log = prepare_ck(m_ck)
    mask_f  = build_tissue_mask(f_log)
    mask_m  = build_tissue_mask(m_log)

    # Log-normalised (AKAZE, NCC, visual QC)
    f_norm      = norm01(f_log.astype(np.float32), mask_f)
    m_raw_norm  = norm01(m_log.astype(np.float32), mask_m)
    # Linear-normalised (RoMaV2 input) — computed here so figs can show them
    f_lin_norm  = norm01(f_lin.astype(np.float32), mask_f)
    m_lin_norm  = norm01(m_lin.astype(np.float32), mask_m)

    # ── AKAZE affine (log images + tissue masks) ─────────────────────────────
    print("Computing AKAZE Affine...")
    M_affine, kp1, kp2, good, inlier_mask = akaze_affine(f_log, m_log, mask_f, mask_m)
    if M_affine is None:
        sys.exit("Affine failed.")

    # Apply affine to lin and log — matching pipeline behaviour
    m_lin_affine  = cv2.warpAffine(m_lin, M_affine, (tw, th), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m_log_affine  = cv2.warpAffine(m_log, M_affine, (tw, th), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_m_affine = cv2.warpAffine(mask_m, M_affine, (tw, th), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m_aff_norm     = norm01(m_log_affine.astype(np.float32), mask_m_affine)
    m_lin_aff_norm = norm01(m_lin_affine.astype(np.float32), mask_m_affine)

    # ── RoMaV2 (linear images, tissue mask applied at LR before upsample) ────
    print("Computing RoMaV2 Dense Warp...")
    map_x, map_y, conf_full = romav2_dense_warp(
        f_lin, m_lin_affine, th, tw, mask_f
    )

    # Apply warp to log-affine (for overlay visualisation — consistent with NCC measurement)
    m_warp     = cv2.remap(m_log_affine, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_m_warp = cv2.remap(mask_m_affine, map_x, map_y,
                             interpolation=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m_warp_norm = norm01(m_warp.astype(np.float32), mask_m_warp)

    m_lin_warp = cv2.remap(m_lin_affine, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_m_lin_warp = cv2.remap(mask_m_affine, map_x, map_y,
                                interpolation=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m_lin_warp_norm = norm01(m_lin_warp.astype(np.float32), mask_m_lin_warp)

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\nGenerating Figures...")
    base = f"{args.fixed_id}_vs_{args.moving_id}"

    fig1_preprocessing_comparison(
        os.path.join(OUT_DIR, f"fig1_preprocessing_{base}.png"),
        f_ck_raw, f_lin, f_log, mask_f
    )
    print(" -> fig1_preprocessing_comparison saved.")

    fig2_affine_stages(
        os.path.join(OUT_DIR, f"fig2_affine_{base}.png"),
        tw, th, f_norm, m_raw_norm, m_aff_norm, f_lin_norm, m_lin_aff_norm,
        mask_f, mask_m, mask_m_affine, kp1, kp2, good, inlier_mask
    )
    print(" -> fig2_affine_stages saved.")

    fig3_romav2_grid(
        os.path.join(OUT_DIR, f"fig3_romav2_grid_{base}.png"),
        tw, th, f_lin_norm, m_lin_aff_norm, f_norm, m_warp_norm,
        mask_f, mask_m_affine, mask_m_warp, map_x, map_y, conf_full
    )
    print(" -> fig3_romav2_grid saved.")

    fig4_warp_field_anatomy(
        os.path.join(OUT_DIR, f"fig4_warp_anatomy_{base}.png"),
        tw, th, 
        f_lin_norm, m_lin_norm, m_lin_aff_norm, m_lin_warp_norm,
        mask_f, mask_m, mask_m_affine, mask_m_lin_warp, 
        map_x, map_y
    )
    print(" -> fig4_warp_field_anatomy saved.")

    fig5_fixed_all_channels(
        os.path.join(OUT_DIR, f"fig5_all_channels_{base}.png"), 
        f_arr, CHANNEL_NAMES
    )
    print(" -> fig5_all_channels saved.")

    print(f"\nSuccess. All plots saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()