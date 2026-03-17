"""
Feature registration — single full-resolution AKAZE affine + INSPIRE elastic.

Pipeline per slice pair:
  L0  AKAZE + ANMS + RANSAC at full resolution -> affine transform
  L1  INSPIRE intensity-based deformation -> elastic residual correction
      Skipped if L0 failed (no valid pre-alignment to refine).

Fallback:
  If L1 fails or diverges (blank image), the pipeline reverts to the L0 affine result.
  If L0 fails, the raw moving slice is written to protect the chain.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import tempfile
import numpy as np
import pandas as pd
import tifffile
import glob
import re
import cv2
import matplotlib
import matplotlib.pyplot as plt
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description='Feature registration — single level full resolution.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
INPUT_FOLDER      = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT       = os.path.join(config.DATASPACE, "Filter_AKAZE_inspire")
OUTPUT_FOLDER     = os.path.join(WORK_OUTPUT, TARGET_CORE)
SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")

# --- INSPIRE CONFIGURATION ---
INSPIRE_REGISTER_BIN  = os.path.join(config.WORKSPACE, "inspire-build", "InspireRegister")
INSPIRE_TRANSFORM_BIN = os.path.join(config.WORKSPACE, "inspire-build", "InspireTransform")
INSPIRE_CONFIG_JSON   = os.path.join(config.WORKSPACE, "tma_configuration.json")

# Dedicated temporary directory on the fast SSD for intermediate TIFF I/O
ACTIVE_WORK_DIR = os.path.join(config.WORKSPACE, "inspire_tmp")

# --- CHANNEL ---
CK_CHANNEL_IDX = 6
CHANNEL_NAMES  = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

# --- OUTPUT PHYSICAL METADATA ---
PIXEL_SIZE_XY_UM     = 0.4961
SECTION_THICKNESS_UM = 4.5

# --- DETECTOR ---
AKAZE_THRESHOLD = 0.0001   # lower threshold — detect more features on weak CK signal
                            # (was 0.0003 in older versions; RANSAC is the quality gate)

# --- ANMS ---
ANMS_KEEP = 6000           # keep more keypoints before matching
                            # More keypoints → more matches → more inliers.
                            # ANMS still enforces spatial spread.

# --- MATCHING ---
LOWE_RATIO  = 0.75         # tighter ratio (compensates for more raw keypoints)
MIN_MATCHES = 20           # minimum matches fed into RANSAC (relaxed for sparse cores)
MIN_INLIERS = 6            # minimum RANSAC survivors to accept the transform

# --- RANSAC ---
RANSAC_CONFIDENCE = 0.995
RANSAC_MAX_ITERS  = 5000
RANSAC_THRESH     = 8.0    # pixels at full resolution

# --- TRANSFORM CONSTRAINTS ---
MAX_SCALE_DEVIATION = 0.08
MAX_SHEAR           = 0.15
MAX_ROTATION_DEG    = 15.0

# --- OUTPUT SANITY ---
MIN_CK_NONZERO_FRAC = 0.01

# INSPIRE must reduce MSE by at least this fraction vs affine-only to be accepted.
# e.g. 0.05 = INSPIRE output must be at least 5% better than affine.
# This catches silent divergence where the optimizer wanders and produces a
# non-blank but distorted output that would poison the next slice.
INSPIRE_MIN_IMPROVEMENT = 0.05

if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ACTIVE_WORK_DIR, exist_ok=True)


# --- SLICE FILTER ---
def load_slice_filter(yaml_path: str, core_name: str):
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path, "r") as fh:
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

# --- UTILITIES ---
def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0

def conform_slice(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    c, h, w = arr.shape
    out    = np.zeros((c, target_h, target_w), dtype=arr.dtype)
    src_y0 = max(0, (h - target_h) // 2)
    dst_y0 = max(0, (target_h - h) // 2)
    copy_h = min(h - src_y0, target_h - dst_y0)
    src_x0 = max(0, (w - target_w) // 2)
    dst_x0 = max(0, (target_w - w) // 2)
    copy_w = min(w - src_x0, target_w - dst_x0)
    out[:, dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
        arr[:, src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
    return out

def prepare_ck(img_arr: np.ndarray):
    """
    Returns (linear_uint8, log_uint8).
    Log image is used for AKAZE detection (boosts weak signal).
    Linear image is used for MSE calculation.
    """
    img_float  = img_arr.astype(np.float32)
    log_img    = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log   = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    p_lo_lin, p_hi_lin = np.percentile(img_arr[::4, ::4], (1, 99.9))
    norm_lin = cv2.normalize(
        np.clip(img_arr, p_lo_lin, p_hi_lin), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return norm_lin, norm_log

def apply_anms(keypoints, descriptors, num_to_keep=2000, c_robust=0.9, max_compute=5000):
    n = len(keypoints)
    if n <= num_to_keep or descriptors is None:
        return keypoints, descriptors
    # max_compute must be at least num_to_keep — otherwise the candidate pool is
    # smaller than the requested output and sort_idx[best] silently wraps/truncates.
    max_compute = max(max_compute, num_to_keep)
    coords    = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])
    sort_idx  = np.argsort(responses)[::-1][:max_compute]
    n         = len(sort_idx)
    coords    = coords[sort_idx]
    responses = responses[sort_idx]
    radii     = np.full(n, np.inf)

    for i in range(1, n):
        stronger = responses[:i] * c_robust > responses[i]
        if np.any(stronger):
            diffs    = coords[:i][stronger] - coords[i]
            radii[i] = np.min(np.sum(diffs ** 2, axis=1))

    best  = np.argsort(radii)[::-1][:num_to_keep]
    final = sort_idx[best]
    return tuple(keypoints[i] for i in final), descriptors[final]

def constrain_affine(M: np.ndarray) -> np.ndarray:
    if M is None: return None
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

# --- L0: AKAZE AFFINE ---
# Uses fixed_log / moving_log (log-normalised uint8) for feature detection —
# log stretch boosts weak CK signal, matching the original design intent.
def akaze_affine(fixed_8bit: np.ndarray, moving_8bit: np.ndarray, slice_id: str):
    detector      = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    kp1_raw, des1 = detector.detectAndCompute(fixed_8bit,  None)
    kp2_raw, des2 = detector.detectAndCompute(moving_8bit, None)

    if des1 is None or des2 is None or len(kp1_raw) < 4 or len(kp2_raw) < 4:
        logger.warning(
            f"[{slice_id}] Feature starvation "
            f"(fixed={len(kp1_raw) if kp1_raw else 0}, moving={len(kp2_raw) if kp2_raw else 0})."
        )
        return None, 0, 0, [], [], [], [], [], np.array([])

    kp1, des1 = apply_anms(kp1_raw, des1, num_to_keep=ANMS_KEEP)
    kp2, des2 = apply_anms(kp2_raw, des2, num_to_keep=ANMS_KEEP)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw if len([m, n]) == 2 and m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_MATCHES:
        logger.warning(f"[{slice_id}] Insufficient matches ({len(good)} < {MIN_MATCHES}).")
        return None, len(good), 0, kp1_raw, kp2_raw, kp1, kp2, good, np.array([])

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(
        dst_pts, src_pts, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH, maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
    )

    if M is None or mask is None:
        logger.warning(f"[{slice_id}] RANSAC diverged.")
        return None, len(good), 0, kp1_raw, kp2_raw, kp1, kp2, good, np.array([])

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        logger.warning(f"[{slice_id}] Inlier count too low ({n_inliers} < {MIN_INLIERS}).")
        return None, len(good), n_inliers, kp1_raw, kp2_raw, kp1, kp2, good, mask

    M = constrain_affine(M)
    if M is None or not transform_is_sane(M):
        U, _, Vt = np.linalg.svd(M[:2, :2])
        R   = U @ Vt
        rot = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        logger.warning(f"[{slice_id}] Transform rejected by sanity gate (rot={rot:.1f}°).")
        return None, len(good), n_inliers, kp1_raw, kp2_raw, kp1, kp2, good, mask

    U, S, _ = np.linalg.svd(M[:2, :2])
    logger.info(
        f"[{slice_id}] AKAZE: matches={len(good)} inliers={n_inliers} "
        f"tx={M[0,2]:.1f}px ty={M[1,2]:.1f}px "
        f"scale={(np.mean(S)-1)*100:+.2f}% shear={(S[0]/S[1]-1)*100:.2f}%"
    )
    return M, len(good), n_inliers, kp1_raw, kp2_raw, kp1, kp2, good, mask

# --- L1: INSPIRE ELASTIC REFINEMENT ---
def apply_inspire_l1(
    fixed_log_for_reg: np.ndarray,
    moving_log_affine: np.ndarray,
    moving_affine_vol: np.ndarray,
    slice_id: str,
):
    """
    Executes INSPIRE L1 elastic registration.

    fixed_log_for_reg  : uint8 log-normalised fixed CK channel  (reference for INSPIRE)
    moving_log_affine  : uint8 log-normalised moving CK channel, already affine-warped
    moving_affine_vol  : float32 full multi-channel volume, already affine-warped
    """
    c, h, w = moving_affine_vol.shape

    with tempfile.TemporaryDirectory(dir=ACTIVE_WORK_DIR, prefix=f"run_{slice_id}_") as temp_dir:
        ref_ck_path = os.path.join(temp_dir, "ref_ck.tif")
        flo_ck_path = os.path.join(temp_dir, "flo_ck.tif")
        tforward    = os.path.join(temp_dir, "tforward.txt")
        treverse    = os.path.join(temp_dir, "treverse.txt")

        # Both images cast to float32 — same numeric scale for the INSPIRE optimizer
        tifffile.imwrite(ref_ck_path, fixed_log_for_reg.astype(np.float32))
        tifffile.imwrite(flo_ck_path, moving_log_affine.astype(np.float32))

        reg_cmd = [
            INSPIRE_REGISTER_BIN, "2",
            "-ref", ref_ck_path,
            "-flo", flo_ck_path,
            "-deform_cfg", INSPIRE_CONFIG_JSON,
            "-out_path_deform_forward", tforward,
            "-out_path_deform_reverse", treverse,
        ]

        try:
            logger.info(f"[{slice_id}] Launching INSPIRE C++ registration...")
            # No capture_output so C++ verbosity prints directly to bash log
            subprocess.run(reg_cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"[{slice_id}] InspireRegister failed with exit code {e.returncode}")
            return None, False

        if not os.path.exists(tforward):
            logger.error(f"[{slice_id}] InspireRegister did not produce transformation map.")
            return None, False

        # Apply the displacement field to every channel of the full 16-bit volume
        aligned_channels = []
        in_channel_path  = os.path.join(temp_dir, "in_channel.tif")
        out_channel_path = os.path.join(temp_dir, "out_channel.tif")

        for i in range(c):
            tifffile.imwrite(in_channel_path, moving_affine_vol[i].astype(np.float32))
            trans_cmd = [
                INSPIRE_TRANSFORM_BIN, "-dim", "2", "-16bit", "1",
                "-interpolation", "linear", "-transform", tforward,
                "-ref", ref_ck_path,
                "-in", in_channel_path,
                "-out", out_channel_path,
            ]
            try:
                subprocess.run(trans_cmd, check=True, capture_output=True)
                aligned_ch = tifffile.imread(out_channel_path)
                aligned_channels.append(aligned_ch)
            except subprocess.CalledProcessError as e:
                logger.error(f"[{slice_id}] InspireTransform failed on ch {i}: {e.stderr}")
                return None, False

        aligned_vol = np.stack(aligned_channels, axis=0).astype(np.uint16)
        logger.info(f"[{slice_id}] INSPIRE elastic refinement complete.")
        return aligned_vol, True

# --- MAIN REGISTRATION PIPELINE ---
def register_slice(fixed_np, moving_np, slice_id=None):
    start = time.time()
    sid   = slice_id or "unknown"

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_lin,  fixed_log  = prepare_ck(fixed_ck)
    moving_lin, moving_log = prepare_ck(moving_ck)
    h, w = fixed_log.shape

    # L0: AKAZE affine — log images (boosts weak signal, original design)
    M_affine, n_matches, n_inliers, kp1_raw, kp2_raw, kp1, kp2, good_matches, inlier_mask = \
        akaze_affine(fixed_log, moving_log, sid)
    akaze_ok = M_affine is not None
    if not akaze_ok:
        M_affine = np.eye(2, 3, dtype=np.float64)

    # Diagnostic plots use the same log images fed to AKAZE
    if len(kp1_raw) > 0:
        save_matching_pairs_plot(
            fixed_log, moving_log, kp1_raw, kp2_raw, kp1, kp2,
            good_matches, inlier_mask, sid, OUTPUT_FOLDER, akaze_ok=akaze_ok,
        )

    # Pre-align every channel with the affine transform
    c = moving_np.shape[0]
    moving_affine_vol = np.zeros_like(moving_np, dtype=np.float32)
    for ch in range(c):
        moving_affine_vol[ch] = cv2.warpAffine(
            moving_np[ch].astype(np.float32), M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    # L1: INSPIRE elastic — log images, affine-prealigned, only when L0 succeeded
    aligned_np = None
    inspire_ok = False
    if akaze_ok:
        moving_log_affine = cv2.warpAffine(
            moving_log, M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        aligned_np, inspire_ok = apply_inspire_l1(
            fixed_log, moving_log_affine, moving_affine_vol, sid,
        )

    # Compute affine-only MSE as the baseline to judge INSPIRE against.
    # Clip to uint16 range first so this is on the same numeric footing as
    # mse_inspire (computed from aligned_np which is already uint16) and the
    # final logged mse.
    affine_ck_u16 = np.clip(moving_affine_vol[CK_CHANNEL_IDX], 0, 65535).astype(np.uint16)
    _, affine_log = prepare_ck(affine_ck_u16.astype(np.float32))
    mse_affine    = float(np.mean((fixed_lin.astype(np.float32) - affine_log.astype(np.float32)) ** 2))

    # Output sanity & fallback
    if inspire_ok:
        ck_out = aligned_np[CK_CHANNEL_IDX]

        # Check 1: blank output (field folded on itself)
        if np.count_nonzero(ck_out) / float(ck_out.size) < MIN_CK_NONZERO_FRAC:
            logger.warning(
                f"[{sid}] CK output nearly blank -- INSPIRE diverged. "
                "Reverting to affine-only result."
            )
            inspire_ok = False
            aligned_np = moving_affine_vol.astype(np.uint16)

        # Check 2: INSPIRE must be meaningfully better than affine alone.
        # "Not worse" is insufficient -- a distorted-but-non-blank output that
        # scores similarly to affine will corrupt the fixed image for the next
        # slice and cascade failures down the chain.
        # Require at least INSPIRE_MIN_IMPROVEMENT fractional reduction in MSE.
        else:
            _, inspire_log = prepare_ck(ck_out.astype(np.float32))
            mse_inspire = float(np.mean((fixed_lin.astype(np.float32) - inspire_log.astype(np.float32)) ** 2))
            threshold   = mse_affine * (1.0 - INSPIRE_MIN_IMPROVEMENT)
            if mse_inspire >= threshold:
                logger.warning(
                    f"[{sid}] INSPIRE MSE ({mse_inspire:.1f}) did not improve enough over "
                    f"affine-only ({mse_affine:.1f}, need <{threshold:.1f}) -- "
                    "reverting to affine-only result."
                )
                inspire_ok = False
                aligned_np = moving_affine_vol.astype(np.uint16)
    else:
        aligned_np = moving_affine_vol.astype(np.uint16)

    # MSE -- linear fixed vs log warped (original metric: cross-domain perceptual measure)
    _, warped_log = prepare_ck(aligned_np[CK_CHANNEL_IDX].astype(np.float32))
    mse = float(np.mean((fixed_lin.astype(np.float32) - warped_log.astype(np.float32)) ** 2))

    # Stats
    U, S, Vt  = np.linalg.svd(M_affine[:2, :2])
    R         = U @ Vt
    rot       = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    scale_pct = (float(np.mean(S)) - 1.0) * 100.0
    shear_pct = (float(S[0] / S[1]) - 1.0) * 100.0 if S[1] > 1e-6 else 0.0

    stats = dict(
        detector     = "AKAZE" if akaze_ok else "Identity",
        n_matches    = n_matches,
        n_inliers    = n_inliers,
        rotation_deg = round(rot, 3),
        tx           = round(float(M_affine[0, 2]), 3),
        ty           = round(float(M_affine[1, 2]), 3),
        scale_pct    = round(scale_pct, 3),
        shear_pct    = round(shear_pct, 3),
        bspline_ok   = inspire_ok,
    )

    return aligned_np, mse, time.time() - start, stats, akaze_ok, M_affine

# --- PLOTTING ---
def _draw_keypoint_marker(canvas, pt, color, radius=10):
    cx, cy = int(pt[0]), int(pt[1])
    cv2.circle(canvas, (cx, cy), radius, color, -1, cv2.LINE_AA)
    arm = radius + 4
    cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 1, cv2.LINE_AA)

def _make_side_by_side_canvas(fixed_8bit, moving_8bit):
    h, w   = fixed_8bit.shape[:2]
    gap    = 6
    canvas = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :w]       = cv2.cvtColor(fixed_8bit,  cv2.COLOR_GRAY2BGR)
    canvas[:, w + gap:] = cv2.cvtColor(moving_8bit, cv2.COLOR_GRAY2BGR)
    return canvas, h, w, gap

def _burn_title(canvas, title, color=(0, 230, 0)):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = max(0.8, canvas.shape[1] / 3000)
    thickness  = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    tx = (canvas.shape[1] - tw) // 2
    ty = th + 10
    cv2.putText(canvas, title, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

def save_matching_pairs_plot(
    fixed_8bit, moving_8bit,
    kp1_raw, kp2_raw, kp1, kp2,
    good_matches, inlier_mask,
    slice_id, output_folder, akaze_ok=True,
):
    out_dir = os.path.join(output_folder, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)
    h, w    = fixed_8bit.shape[:2]
    gap     = 6
    status  = "SUCCESS" if akaze_ok else "FAILED"
    title_color = (0, 230, 0) if akaze_ok else (0, 0, 220)

    canvas_pre, _, _, _ = _make_side_by_side_canvas(fixed_8bit, moving_8bit)
    rng        = np.random.default_rng(seed=0)
    kp1_sample = list(kp1_raw)
    kp2_sample = list(kp2_raw)
    if len(kp1_sample) > 3000:
        kp1_sample = [kp1_sample[i] for i in rng.choice(len(kp1_sample), 3000, replace=False)]
    if len(kp2_sample) > 3000:
        kp2_sample = [kp2_sample[i] for i in rng.choice(len(kp2_sample), 3000, replace=False)]
    for kp in kp1_sample: _draw_keypoint_marker(canvas_pre, kp.pt, (0, 200, 0), radius=5)
    for kp in kp2_sample: _draw_keypoint_marker(canvas_pre, (kp.pt[0] + w + gap, kp.pt[1]), (0, 0, 200), radius=5)
    _burn_title(canvas_pre, f"{slice_id}  kp_fixed={len(kp1_raw)}  kp_moving={len(kp2_raw)}  [BEFORE ANMS]", color=(180, 180, 0))
    cv2.imwrite(os.path.join(out_dir, f"{slice_id}_before_anms.png"), canvas_pre)
    logger.info(f"[{slice_id}] Before-ANMS plot saved.")

    inlier_matches = [m for m, keep in zip(good_matches, inlier_mask.ravel()) if keep] \
                     if len(inlier_mask) > 0 else []
    n_inliers = len(inlier_matches)

    canvas_post, _, _, _ = _make_side_by_side_canvas(fixed_8bit, moving_8bit)
    for idx, m in enumerate(inlier_matches[:200]):
        hue       = int(idx / max(len(inlier_matches[:200]) - 1, 1) * 179)
        color_bgr = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR)[0, 0])
        pt1 = kp1[m.queryIdx].pt
        pt2 = (kp2[m.trainIdx].pt[0] + w + gap, kp2[m.trainIdx].pt[1])
        cv2.line(canvas_post, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color_bgr, 1, cv2.LINE_AA)
        _draw_keypoint_marker(canvas_post, pt1, (0, 200, 0))
        _draw_keypoint_marker(canvas_post, pt2, (0, 0, 200))

    _burn_title(canvas_post, f"{slice_id}  inliers={n_inliers}  [{status}]", color=title_color)
    cv2.imwrite(os.path.join(out_dir, f"{slice_id}_after_anms.png"), canvas_post)
    logger.info(f"[{slice_id}] After-ANMS plot saved.")

def generate_qc_montage(
    vol, output_folder,
    slice_ids=None, channel_idx=6,
    channel_name="CK", title_suffix="AKAZE+INSPIRE",
):
    logger.info(f"Generating QC montage for {channel_name} channel ({title_suffix}).")
    n_slices = vol.shape[0]
    if n_slices < 2:
        logger.warning("Fewer than 2 slices — QC montage skipped.")
        return

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

        def norm(x):
            p99 = np.percentile(x, 99.5)
            return np.clip(x / (p99 if p99 > 0 else 1), 0, 1)

        overlay = np.dstack((norm(s1), norm(s2), np.zeros_like(s1)))
        axes_flat[idx].imshow(overlay)

        lbl1 = slice_ids[z1] if slice_ids is not None else z1
        lbl2 = slice_ids[z2] if slice_ids is not None else z2
        axes_flat[idx].set_title(f"ID{lbl1} to ID{lbl2}", fontsize=10, fontweight='bold')
        axes_flat[idx].axis('off')

    for idx in range(len(all_pairs), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.suptitle(f'Registration QC {title_suffix}: {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(
        output_folder,
        f"{TARGET_CORE}_QC_Montage_{channel_name}_{title_suffix}.png",
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")

# --- MAIN EXECUTION ---
def main():
    logger.info(f"Feature Registration (AKAZE + INSPIRE) — {TARGET_CORE}")
    logger.info(
        f"AKAZE threshold={AKAZE_THRESHOLD} | ANMS keep={ANMS_KEEP} | "
        f"Lowe ratio={LOWE_RATIO} | Min matches={MIN_MATCHES} | "
        f"Min inliers={MIN_INLIERS} | RANSAC thresh={RANSAC_THRESH}px | INSPIRE min improvement={INSPIRE_MIN_IMPROVEMENT*100:.0f}%"
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
        file_list = [f for i, f in enumerate(file_list) if i in allowed_positions]
        n_slices  = len(file_list)
        excluded  = original_count - n_slices
        logger.info(
            f"Slice filter active for {TARGET_CORE}: "
            f"keeping {n_slices}/{original_count} slices "
            f"(positions {sorted(allowed_positions)}), "
            f"{excluded} excluded."
        )
        if n_slices == 0:
            logger.error("Slice filter excluded all slices — check slice_filter.yaml.")
            sys.exit(1)
    else:
        logger.info(f"No slice filter for {TARGET_CORE} — using all {n_slices} slices.")

    if n_slices < 2:
        logger.warning("Only one slice — writing identity output.")
        vol_in   = tifffile.imread(file_list[0])
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
        tifffile.imwrite(
            out_path, vol_in[np.newaxis],
            photometric='minisblack',
            metadata={
                'axes': 'ZCYX',
                'Channel': {'Name': CHANNEL_NAMES},
                'PhysicalSizeX': PIXEL_SIZE_XY_UM,
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': PIXEL_SIZE_XY_UM,
                'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZ': SECTION_THICKNESS_UM,
                'PhysicalSizeZUnit': 'µm',
            },
            compression='deflate', compressionargs={'level': 6},
        )
        sys.exit(0)

    center_idx  = n_slices // 2
    _center_arr = tifffile.imread(file_list[center_idx])
    if _center_arr.ndim == 2:
        _center_arr = _center_arr[np.newaxis]
    elif _center_arr.ndim == 3 and _center_arr.shape[-1] < _center_arr.shape[0]:
        _center_arr = np.moveaxis(_center_arr, -1, 0)

    c, target_h, target_w = _center_arr.shape
    logger.info(f"Canonical full-res shape: C={c}, H={target_h}, W={target_w}")
    logger.info(f"Loading and conforming {n_slices} slices.")

    raw_slices = []
    slice_ids  = []
    for f in file_list:
        slice_ids.append(get_slice_number(f))
        arr = tifffile.imread(f)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)
        if arr.shape[1] != target_h or arr.shape[2] != target_w:
            logger.warning(f"Shape mismatch in {os.path.basename(f)} — conforming.")
            arr = conform_slice(arr, target_h, target_w)
        raw_slices.append(arr)

    aligned_vol             = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    aligned_vol[center_idx] = raw_slices[center_idx]
    anchor_id               = get_slice_number(file_list[center_idx])
    logger.info(f"Volume anchored at slice index {center_idx} (ID {anchor_id})")

    registration_stats = []

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} pass.")

        for i in indices:
            real_id   = get_slice_number(file_list[i])
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = raw_slices[i]
            sid       = f"Z{i:03d}_ID{real_id:03d}"

            aligned_np, mse, runtime, stats, success, M_final = register_slice(
                fixed_np, moving_np, slice_id=sid,
            )

            if success and transform_is_sane(M_final):
                aligned_vol[i] = aligned_np
                status_str     = "SUCCESS"
            else:
                aligned_vol[i] = raw_slices[i]
                status_str     = "IDENTITY_FALLBACK_RAW"
                logger.warning(f"Z{i:02d} (ID {real_id:03d}): {status_str} — writing raw slice.")

            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | Det: {stats['detector']} | "
                f"Matches: {stats['n_matches']} | Inliers: {stats['n_inliers']} | "
                f"INSPIRE: {stats['bspline_ok']} | MSE: {mse:.2f} | "
                f"Rot: {stats['rotation_deg']:.2f} | "
                f"tx: {stats['tx']:.1f}px | ty: {stats['ty']:.1f}px | "
                f"Scale: {stats['scale_pct']:+.2f}% | Shear: {stats['shear_pct']:.2f}% | "
                f"t: {runtime:.2f}s | Status: {status_str}"
            )

            registration_stats.append({
                "Direction":    direction,
                "Slice_Z":      i,
                "Slice_ID":     real_id,
                "Detector":     stats["detector"],
                "N_Matches":    stats["n_matches"],
                "N_Inliers":    stats["n_inliers"],
                "BSpline_OK":   stats["bspline_ok"],
                "Success":      success,
                "Status":       status_str,
                "MSE_After":    round(mse, 4),
                "Rotation_Deg": stats["rotation_deg"],
                "Shift_X_px":   stats["tx"],
                "Shift_Y_px":   stats["ty"],
                "Scale_Pct":    stats["scale_pct"],
                "Shear_Pct":    stats["shear_pct"],
                "Runtime_s":    round(runtime, 3),
            })

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx + 1, n_slices), "Forward")

    df   = pd.DataFrame(registration_stats).sort_values("Slice_Z")
    cols = [
        "Direction", "Slice_Z", "Slice_ID", "Detector",
        "N_Matches", "N_Inliers", "BSpline_OK",
        "Success", "Status", "Rotation_Deg",
        "Shift_X_px", "Shift_Y_px", "Scale_Pct", "Shear_Pct",
        "MSE_After", "Runtime_s",
    ]
    df[cols].to_csv(
        os.path.join(OUTPUT_FOLDER, "registration_stats_AKAZE_INSPIRE.csv"), index=False
    )

    n_ok       = int((df["Status"] == "SUCCESS").sum())
    n_fallback = int((df["Status"] == "IDENTITY_FALLBACK_RAW").sum())
    logger.info(
        f"Execution complete. SUCCESS: {n_ok} | IDENTITY_FALLBACK_RAW: {n_fallback}"
    )

    # --- QC MONTAGES (generated before writing the volume) ---
    logger.info("Generating QC montages...")
    raw_vol = np.stack(raw_slices, axis=0)
    generate_qc_montage(
        raw_vol, OUTPUT_FOLDER,
        slice_ids=slice_ids,
        channel_idx=CK_CHANNEL_IDX,
        channel_name="CK",
        title_suffix="AKAZE_Raw",
    )
    generate_qc_montage(
        aligned_vol, OUTPUT_FOLDER,
        slice_ids=slice_ids,
        channel_idx=CK_CHANNEL_IDX,
        channel_name="CK",
        title_suffix="AKAZE+INSPIRE",
    )

    # --- WRITE REGISTERED VOLUME ---
    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
    logger.info(f"Writing registered volume to {out_tiff}")
    tifffile.imwrite(
        out_tiff, aligned_vol,
        photometric='minisblack',
        metadata={
            'axes': 'ZCYX',
            'Channel': {'Name': CHANNEL_NAMES},
            'PhysicalSizeX': PIXEL_SIZE_XY_UM,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': PIXEL_SIZE_XY_UM,
            'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeZ': SECTION_THICKNESS_UM,
            'PhysicalSizeZUnit': 'µm',
        },
        compression='deflate', compressionargs={'level': 6},
    )
    logger.info("Done.")

if __name__ == "__main__":
    main()