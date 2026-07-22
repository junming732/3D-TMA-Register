"""
Diagnostics for flagged 'unsuitable' cores.
============================================

For a small set of cores that have been flagged as problematic (default:
16, 17, 21, 23, 27), this script analyzes candidate images per slice
(default set, all seven): DAPI, CK, AF (each log-normalised individually),
'af_linear' (AF with a straight percentile stretch, no log1p — log
compresses differences most at high values, which is exactly where AF's
real signal is bunched, so this variant tests whether skipping it helps),
'fusion' (DAPI+AF+CK weighted-blend collapsed to gray, AF still log-
normalised — same recipe as prepare_3ch_fusion in
akaze_romav2_multi_channel_warp.py), 'fusion_af_linear' (identical fusion,
but AF uses the same linear stretch as af_linear instead of log — isolates
whether the af_linear finding also helps once AF is folded into the fusion
composite, not just alone), 'fusion_equal' (identical RGB composite to
'fusion', but collapsed to gray via a simple equal-weighted channel mean
instead of cv2.cvtColor's default RGB2GRAY luma weights — those weights
are calibrated for human colour perception and, applied to DAPI->R/AF->G/
CK->B, weight AF almost 2x DAPI and ~5x CK, which has no particular
justification for these channels; this candidate tests whether that
default weighting is actually helping or hurting), 'color_lut' (7-channel,
AF-excluded, colour-LUT weighted blend collapsed to gray — same recipe as
prepare_color_lut_fusion in that script), and 'color_lut_equal' (same idea
as fusion_equal, applied to the color_lut composite instead). 'dapi_linear'
and 'ck_linear' are also available via --channels if you want them. For
each selected candidate, it:

  1. Builds a per-channel contact-sheet montage across every slice in the
     core, so you can visually scan how each candidate looks slice-by-slice
     — useful for spotting exactly where signal drops out.

  2. For every adjacent slice pair, builds a magenta/green overlay per
     candidate (fixed=magenta, moving=green; white/grey = well aligned) plus
     a combined side-by-side figure (one panel per selected candidate) for
     that pair, so misalignment is visible before any registration is
     attempted.

  3. Runs AKAZE (masked, same detector/matcher/RANSAC settings as the L0
     channel-comparison script) independently on every selected candidate
     for every adjacent pair, and records N_Matches / N_Inliers / success.
     Also measures masked NCC on CK log before and after each candidate's
     own estimated affine (same reference channel for every candidate, so
     they're all scored against the same real-alignment yardstick — a
     candidate can produce a geometrically 'successful' transform with few
     inliers that still doesn't actually improve tissue alignment, which
     match/inlier counts alone wouldn't catch). Plus an inlier-match
     visualization — so you can see, per problem core, which candidate (if
     any) actually has enough keypoints to register on, and whether doing
     so actually helped.

Tissue masks are loaded from the precomputed '<stem>_tissue_mask.png'
sibling files (see crop_conform_mask_tma.py); if one is missing for a given
slice, matching falls back to no masking for that slice with a warning
(better to still show you the result than silently skip a problem core).

Usage:
    python analyze_unsuitable_cores.py
    python analyze_unsuitable_cores.py --core_ids 16,17,21,23,27
    python analyze_unsuitable_cores.py --core_ids 16 --channels dapi,ck,af_linear
    python analyze_unsuitable_cores.py --core_ids 1-30
    python analyze_unsuitable_cores.py --core_ids 1-15,18-30   # ranges + individual, mixed
"""

import os
import sys
import glob
import re
import logging
import argparse

import numpy as np
import pandas as pd
import cv2
import tifffile
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Diagnostics for flagged unsuitable cores.")
parser.add_argument('--core_ids', type=str, default="16,17,21,23,27",
                    help="Comma-separated core numbers and/or 'lo-hi' ranges, "
                         "e.g. '16,17,21,23,27' or '1-30' or '1-15,18-30'.")
parser.add_argument('--channels', type=str,
                    default="dapi,ck,af,af_linear,fusion,fusion_af_linear,fusion_equal,"
                            "color_lut,color_lut_equal",
                    help="Comma-separated candidates: dapi,ck,af,af_linear,ck_linear,"
                         "dapi_linear,fusion,fusion_af_linear,fusion_equal,"
                         "color_lut,color_lut_equal")
args = parser.parse_args()

def parse_core_ids(spec: str) -> list:
    """
    Parse a comma-separated core-ID spec supporting individual numbers and
    'lo-hi' ranges, e.g. '1-15,18-30' or '16,17,21,23,27'. Same range syntax
    already used for slice_filter.yaml elsewhere in this pipeline. Preserves
    the order given (ranges expand in order, no dedup/sort — repeat entries
    are the caller's responsibility to avoid).
    """
    ids = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            ids.extend(range(int(lo.strip()), int(hi.strip()) + 1))
        else:
            ids.append(int(part))
    return ids


CORE_IDS = parse_core_ids(args.core_ids)
CORE_NAMES = [f"Core_{n:02d}" for n in CORE_IDS]

DAPI_CHANNEL_IDX = 0
CK_CHANNEL_IDX   = 6
AF_CHANNEL_IDX   = 7
CHANNEL_IDX = {
    'dapi': DAPI_CHANNEL_IDX, 'ck': CK_CHANNEL_IDX, 'af': AF_CHANNEL_IDX,
    'dapi_linear': DAPI_CHANNEL_IDX, 'ck_linear': CK_CHANNEL_IDX, 'af_linear': AF_CHANNEL_IDX,
}
LINEAR_CANDIDATES = {'dapi_linear', 'ck_linear', 'af_linear'}
CHANNEL_LABEL = {
    'dapi': 'DAPI', 'ck': 'CK', 'af': 'AF',
    'dapi_linear': 'DAPI (linear, no log)', 'ck_linear': 'CK (linear, no log)',
    'af_linear': 'AF (linear, no log)',
    'fusion': 'Fusion (DAPI+AF+CK)', 'fusion_af_linear': 'Fusion (AF linear)',
    'fusion_equal': 'Fusion (equal-weight gray)',
    'color_lut': 'Color LUT (7ch)', 'color_lut_equal': 'Color LUT (equal-weight gray)',
}
VALID_CANDIDATES = {'dapi', 'ck', 'af', 'dapi_linear', 'ck_linear', 'af_linear',
                    'fusion', 'fusion_af_linear', 'fusion_equal',
                    'color_lut', 'color_lut_equal'}

# Same 7-index colour LUT as akaze_romav2_multi_channel_warp.py (AF excluded)
COLOR_LUT = {
    0: (0,   128, 255),
    1: (51,  255,  51),
    2: (255,  51,  51),
    3: (0,   255, 255),
    4: (255,   0, 255),
    5: (255, 255,   0),
    6: (255, 128,   0),
}
FUSION_WEIGHTS_NOTE = "DAPI->R, AF->G, CK->B (matches prepare_3ch_fusion in the RoMaV2 script)"

CHANNELS = [c.strip().lower() for c in args.channels.split(',') if c.strip()]
for c in CHANNELS:
    if c not in VALID_CANDIDATES:
        raise ValueError(f"Unknown candidate '{c}' — must be one of {sorted(VALID_CANDIDATES)}.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
OUT_BASE       = os.path.join(config.DATASPACE, "Unsuitable_Core_Diagnostics")
os.makedirs(OUT_BASE, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# AKAZE CONFIG — same as akaze_l0_channel_comparison.py, for comparable results
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


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def get_mask_sibling_path(tif_path: str) -> str:
    if tif_path.endswith(".ome.tif"):
        stem = tif_path[:-len(".ome.tif")]
    else:
        stem = os.path.splitext(tif_path)[0]
    return stem + "_tissue_mask.png"


def load_as_chw(path: str) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.moveaxis(arr, -1, 0)
    return arr


def log_normalize(channel_f32, lo=0.1, hi=99.9):
    log_img    = np.log1p(channel_f32)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (lo, hi))
    return cv2.normalize(np.clip(log_img, p_lo, p_hi), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)


def linear_normalize(channel_f32, lo=0.1, hi=99.9):
    """No log1p — straight percentile stretch. Matches prepare_ck's norm_lin
    exactly (same recipe, generalised to any channel). Log compresses
    differences most at high values, which is exactly the range AF's real
    signal is bunched into — this skips that, so contrast that log squashes
    stays intact."""
    p_lo, p_hi = np.percentile(channel_f32[::4, ::4], (lo, hi))
    return cv2.normalize(np.clip(channel_f32, p_lo, p_hi), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)


def _prepare_single(img_arr, lo=0.1, hi=99.5):
    """Matches akaze_romav2_multi_channel_warp.py's _prepare_single exactly
    (note: hi=99.5 here, vs 99.9 in log_normalize above — kept distinct so
    fusion/color_lut candidates are pixel-for-pixel identical to what the
    RoMaV2 script actually feeds it, not just 'close enough')."""
    img_float = img_arr.astype(np.float32)
    log_img   = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (lo, hi))
    return cv2.normalize(np.clip(log_img, p_lo, p_hi), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)


def prepare_3ch_fusion(vol):
    """(H, W, 3) uint8 RGB: DAPI->R, AF->G, CK->B, each log-normalised independently.
    Identical recipe to prepare_3ch_fusion in akaze_romav2_multi_channel_warp.py."""
    dapi = _prepare_single(vol[DAPI_CHANNEL_IDX])
    af   = _prepare_single(vol[AF_CHANNEL_IDX])
    ck   = _prepare_single(vol[CK_CHANNEL_IDX])
    return np.stack([dapi, af, ck], axis=-1)


def prepare_3ch_fusion_af_linear(vol):
    """Same as prepare_3ch_fusion, except AF uses linear_normalize (no log1p)
    instead of _prepare_single — DAPI and CK are untouched, so AF's
    normalization is the only variable changed. Tests whether the af_linear
    finding (log over-compresses AF's already-narrow contrast) also helps
    once AF is folded into the 3-channel fusion composite, not just alone."""
    dapi = _prepare_single(vol[DAPI_CHANNEL_IDX])
    af   = linear_normalize(vol[AF_CHANNEL_IDX].astype(np.float32))
    ck   = _prepare_single(vol[CK_CHANNEL_IDX])
    return np.stack([dapi, af, ck], axis=-1)


def prepare_color_lut_fusion(vol):
    """(H, W, 3) uint8 RGB: 7-channel (AF excluded) weighted-average colour-LUT
    blend. Identical recipe to prepare_color_lut_fusion in
    akaze_romav2_multi_channel_warp.py."""
    h, w = vol.shape[1], vol.shape[2]
    acc  = np.zeros((h, w, 3), dtype=np.float32)
    n    = len(COLOR_LUT)
    for idx, color in COLOR_LUT.items():
        norm      = _prepare_single(vol[idx]).astype(np.float32) / 255.0
        color_arr = np.array(color, dtype=np.float32) / 255.0
        acc      += norm[..., None] * color_arr[None, None, :]
    return np.clip(acc / n * 255.0, 0, 255).astype(np.uint8)


def get_display_and_gray(channel_key, vol):
    """
    Returns (display_img, gray_img) for a candidate:
      - dapi/ck/af           -> both are the same grayscale log-normalised channel.
      - fusion/fusion_equal  -> display is the colour RGB composite; gray is its
                                grayscale flattening, used for AKAZE/overlay.
      - color_lut/_equal     -> same idea, RGB colour-LUT composite.
    AKAZE needs a single channel, so gray is always what gets matched on;
    display is only for the montage/overlay figures.

    The '_equal' variants use the identical RGB composite as their non-'_equal'
    counterpart, differing only in how it's collapsed to gray: a plain
    per-channel mean instead of cv2.cvtColor's default RGB2GRAY luma weights
    (0.299/0.587/0.114) — weights calibrated for human colour perception,
    with no particular justification for DAPI/AF/CK channels mapped into
    R/G/B slots. Kept as separate, directly-comparable candidates rather
    than a global toggle, consistent with how fusion vs fusion_af_linear
    isolates one variable at a time.
    """
    if channel_key in CHANNEL_IDX:
        idx = CHANNEL_IDX[channel_key]
        if channel_key in LINEAR_CANDIDATES:
            gray = linear_normalize(vol[idx].astype(np.float32))
        else:
            gray = log_normalize(vol[idx].astype(np.float32))
        return gray, gray
    if channel_key in ('fusion', 'fusion_equal'):
        rgb = prepare_3ch_fusion(vol)
    elif channel_key == 'fusion_af_linear':
        rgb = prepare_3ch_fusion_af_linear(vol)
    elif channel_key in ('color_lut', 'color_lut_equal'):
        rgb = prepare_color_lut_fusion(vol)
    else:
        raise ValueError(channel_key)
    if channel_key in ('fusion_equal', 'color_lut_equal'):
        gray = rgb.astype(np.float32).mean(axis=2).astype(np.uint8)
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def load_mask_or_none(tif_path: str, shape_hw):
    mask_path = get_mask_sibling_path(tif_path)
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == shape_hw:
            return mask
        logger.warning(f"Mask at {mask_path} missing/shape-mismatched — matching unmasked.")
    else:
        logger.warning(f"No precomputed mask for {os.path.basename(tif_path)} — matching unmasked.")
    return None


def build_overlay(fixed_img_u8, moving_img_u8):
    """Magenta (fixed) / green (moving) overlay. White-ish where well aligned."""
    overlay = np.zeros((*fixed_img_u8.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = fixed_img_u8    # B
    overlay[:, :, 1] = moving_img_u8   # G
    overlay[:, :, 2] = fixed_img_u8    # R
    return overlay


def measure_ncc(fixed_f32: np.ndarray, moving_f32: np.ndarray, mask_uint8) -> float:
    """Masked NCC via 0-iteration SimpleITK LBFGSB (evaluation only, no
    optimisation). More negative = better. Always measured on CK log,
    regardless of which candidate produced the transform being scored —
    keeps every candidate answerable to the same real-alignment yardstick."""
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
# AKAZE AFFINE — same recipe as akaze_l0_channel_comparison.py
# ─────────────────────────────────────────────────────────────────────────────

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


def akaze_affine(fixed_img, moving_img, slice_id, fixed_mask, moving_mask):
    detector = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    kp1_raw, des1 = detector.detectAndCompute(fixed_img,  fixed_mask)
    kp2_raw, des2 = detector.detectAndCompute(moving_img, moving_mask)
    n1 = len(kp1_raw) if kp1_raw else 0
    n2 = len(kp2_raw) if kp2_raw else 0

    if des1 is None or des2 is None or n1 < 4 or n2 < 4:
        logger.info(f"[{slice_id}] Feature starvation (fixed={n1}, moving={n2}).")
        return None, 0, 0, [], [], [], np.array([])

    def cap_by_response(kps, des, max_kp):
        if len(kps) <= max_kp:
            return kps, des
        idx = np.argsort([kp.response for kp in kps])[::-1][:max_kp]
        return tuple(kps[i] for i in idx), des[idx]

    kp1, des1 = cap_by_response(kp1_raw, des1, AKAZE_MAX_KEYPOINTS)
    kp2, des2 = cap_by_response(kp2_raw, des2, AKAZE_MAX_KEYPOINTS)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        logger.info(f"[{slice_id}] Insufficient matches ({len(good)} < {MIN_MATCHES}).")
        return None, len(good), 0, kp1, kp2, good, np.array([])

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC,
                                   ransacReprojThreshold=RANSAC_THRESH,
                                   maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE)
    if M is None or mask is None:
        logger.info(f"[{slice_id}] RANSAC diverged.")
        return None, len(good), 0, kp1, kp2, good, np.array([])

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        logger.info(f"[{slice_id}] Inlier count too low ({n_inliers} < {MIN_INLIERS}).")
        return None, len(good), n_inliers, kp1, kp2, good, mask

    M = constrain_affine(M)
    if M is None or not transform_is_sane(M):
        logger.info(f"[{slice_id}] Transform rejected (rotation out of range).")
        return None, len(good), n_inliers, kp1, kp2, good, mask

    return M, len(good), n_inliers, kp1, kp2, good, mask


def save_inlier_plot(out_path, fixed_img, moving_img, kp1, kp2, good_matches,
                     inlier_mask, title, akaze_ok):
    h, w   = fixed_img.shape
    gap    = 6
    canvas = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :w]       = cv2.cvtColor(fixed_img,  cv2.COLOR_GRAY2BGR)
    canvas[:, w + gap:] = cv2.cvtColor(moving_img, cv2.COLOR_GRAY2BGR)

    inlier_matches = ([m for m, keep in zip(good_matches, inlier_mask.ravel()) if keep]
                      if len(inlier_mask) > 0 else [])
    for idx, m in enumerate(inlier_matches[:200]):
        hue       = int(idx / max(len(inlier_matches[:200]) - 1, 1) * 179)
        color_bgr = tuple(int(c) for c in
                          cv2.cvtColor(np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR)[0, 0])
        pt1 = kp1[m.queryIdx].pt
        pt2 = (kp2[m.trainIdx].pt[0] + w + gap, kp2[m.trainIdx].pt[1])
        cv2.line(canvas, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                 color_bgr, 1, cv2.LINE_AA)

    status      = "SUCCESS" if akaze_ok else "FAILED"
    title_color = (0, 230, 0) if akaze_ok else (0, 0, 220)
    full_title  = f"{title}  inliers={len(inlier_matches)}  [{status}]"
    font        = cv2.FONT_HERSHEY_SIMPLEX
    scale       = max(0.8, canvas.shape[1] / 3000)
    thickness   = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(full_title, font, scale, thickness)
    cv2.putText(canvas, full_title, ((canvas.shape[1] - tw) // 2, th + 10),
                font, scale, title_color, thickness, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)


# ─────────────────────────────────────────────────────────────────────────────
# PER-CORE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def build_channel_montage(core_name, channel_key, file_list, slice_ids, out_dir):
    """One contact-sheet PNG per candidate: every slice, labeled. Grayscale
    channels (dapi/ck/af) render with cmap='gray'; fusion/color_lut render
    as their actual RGB composite."""
    n = len(file_list)
    n_cols = min(6, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    is_rgb = channel_key in ('fusion', 'fusion_af_linear', 'fusion_equal',
                             'color_lut', 'color_lut_equal')

    for i, (f, sid) in enumerate(zip(file_list, slice_ids)):
        vol = load_as_chw(f)
        display_img, _ = get_display_and_gray(channel_key, vol)
        axes[i].imshow(display_img) if is_rgb else axes[i].imshow(display_img, cmap='gray')
        axes[i].set_title(f"Z{i:02d} (ID {sid})", fontsize=9)
        axes[i].axis('off')
    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"{core_name} — {CHANNEL_LABEL[channel_key]}", fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{core_name}_{channel_key}_montage.png")
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"[{core_name}] Saved {CHANNEL_LABEL[channel_key]} montage -> {out_path}")


def process_core(core_name):
    input_dir = os.path.join(DATA_BASE_PATH, core_name)
    file_list = sorted(glob.glob(os.path.join(input_dir, "*.ome.tif")), key=get_slice_number)
    if len(file_list) == 0:
        logger.warning(f"[{core_name}] No .ome.tif files found — skipping.")
        return []
    slice_ids = [get_slice_number(f) for f in file_list]

    out_dir      = os.path.join(OUT_BASE, core_name)
    overlay_dir  = os.path.join(out_dir, "overlays")
    inlier_dir   = os.path.join(out_dir, "akaze_inliers")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(inlier_dir, exist_ok=True)

    logger.info(f"=== [{core_name}] {len(file_list)} slices — channels: {CHANNELS} ===")

    # --- Step 1: per-channel montage across all slices ---
    for ch in CHANNELS:
        build_channel_montage(core_name, ch, file_list, slice_ids, out_dir)

    rows = []
    n = len(file_list)
    if n < 2:
        logger.warning(f"[{core_name}] Fewer than 2 slices — skipping pairwise steps.")
        return rows

    # --- Step 2 & 3: pairwise overlays + AKAZE stats, per channel ---
    for i in range(n - 1):
        fixed_path,  moving_path  = file_list[i + 1], file_list[i]
        fixed_sid,   moving_sid   = slice_ids[i + 1],  slice_ids[i]
        sid = f"Z{i:02d}_ID{moving_sid}_to_Z{i+1:02d}_ID{fixed_sid}"

        fixed_vol  = load_as_chw(fixed_path)
        moving_vol = load_as_chw(moving_path)
        h, w = fixed_vol.shape[1:]

        fixed_mask  = load_mask_or_none(fixed_path,  (h, w))
        moving_mask = load_mask_or_none(moving_path, (h, w))

        # NCC reference — always CK log, regardless of candidate, so every
        # candidate's transform is scored against the same real-alignment
        # yardstick (see measure_ncc docstring).
        fixed_ck_log  = log_normalize(fixed_vol[CK_CHANNEL_IDX].astype(np.float32))
        moving_ck_log = log_normalize(moving_vol[CK_CHANNEL_IDX].astype(np.float32))
        ncc_raw = measure_ncc(fixed_ck_log.astype(np.float32),
                              moving_ck_log.astype(np.float32), fixed_mask)

        pair_overlays = {}
        for ch in CHANNELS:
            _, fixed_img  = get_display_and_gray(ch, fixed_vol)
            _, moving_img = get_display_and_gray(ch, moving_vol)

            # Overlay (raw, unregistered — shows misalignment as-is)
            overlay = build_overlay(fixed_img, moving_img)
            cv2.imwrite(os.path.join(overlay_dir, f"{sid}_{ch}_overlay.png"), overlay)
            pair_overlays[ch] = overlay

            # AKAZE match/inlier stats
            M, n_matches, n_inliers, kp1, kp2, good, inlier_mask = akaze_affine(
                fixed_img, moving_img, f"{sid}_{ch}", fixed_mask, moving_mask
            )
            akaze_ok = M is not None

            # NCC before/after THIS candidate's own estimated affine, always
            # measured on CK log — answers "did detecting on this candidate
            # actually improve real tissue alignment?", not just "did AKAZE
            # find a geometrically-consistent transform?"
            if akaze_ok:
                moving_ck_log_affine = cv2.warpAffine(
                    moving_ck_log, M, (w, h),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                ncc_affine = measure_ncc(fixed_ck_log.astype(np.float32),
                                         moving_ck_log_affine.astype(np.float32), fixed_mask)
            else:
                ncc_affine = ncc_raw
            improvement_pct = (
                (ncc_raw - ncc_affine) / abs(ncc_raw) * 100.0 if abs(ncc_raw) > 1e-9 else 0.0
            )

            save_inlier_plot(
                os.path.join(inlier_dir, f"{sid}_{ch}_inliers.png"),
                fixed_img, moving_img, kp1, kp2, good, inlier_mask,
                f"{core_name} {sid} [{CHANNEL_LABEL[ch]}]", akaze_ok
            )
            rows.append(dict(
                Core=core_name, Slice_Pair=sid, Channel=CHANNEL_LABEL[ch],
                AKAZE_OK=akaze_ok, N_Matches=n_matches, N_Inliers=n_inliers,
                NCC_Raw=round(float(ncc_raw), 6), NCC_Affine=round(float(ncc_affine), 6),
                NCC_Improvement_Pct=round(float(improvement_pct), 2),
            ))
            logger.info(f"[{core_name}][{sid}][{CHANNEL_LABEL[ch]}] "
                       f"ok={akaze_ok} matches={n_matches} inliers={n_inliers} "
                       f"NCC {ncc_raw:.4f}->{ncc_affine:.4f} ({improvement_pct:+.1f}%)")

        # Combined overlay figure for this pair — one panel per selected candidate
        fig, axes = plt.subplots(1, len(CHANNELS), figsize=(5 * len(CHANNELS), 5))
        axes = np.atleast_1d(axes)
        for ax, ch in zip(axes, CHANNELS):
            ax.imshow(cv2.cvtColor(pair_overlays[ch], cv2.COLOR_BGR2RGB))
            ax.set_title(CHANNEL_LABEL[ch])
            ax.axis('off')
        fig.suptitle(f"{core_name} — {sid} (magenta=fixed, green=moving)", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, f"{sid}_combined_overlay.png"), dpi=130)
        plt.close(fig)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Analyzing flagged cores: {CORE_NAMES}")
    all_rows = []
    for core_name in CORE_NAMES:
        all_rows.extend(process_core(core_name))

    if not all_rows:
        logger.error("No results produced for any flagged core.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_BASE, "unsuitable_cores_akaze_summary.csv")
    df.to_csv(csv_path, index=False)

    summary = df.groupby(['Core', 'Channel']).agg(
        n_pairs=('AKAZE_OK', 'size'),
        success_rate_pct=('AKAZE_OK', lambda s: round(100 * s.mean(), 1)),
        mean_inliers=('N_Inliers', 'mean'),
        mean_ncc_improv_pct=('NCC_Improvement_Pct', 'mean'),
    ).reset_index()
    logger.info("Per-core, per-channel summary:\n" + summary.to_string(index=False))
    summary.to_csv(os.path.join(OUT_BASE, "unsuitable_cores_summary_by_channel.csv"), index=False)

    logger.info(f"Done. Results in {OUT_BASE}")


if __name__ == "__main__":
    main()