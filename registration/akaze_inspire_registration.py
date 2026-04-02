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
LOWE_RATIO  = 0.8         # tighter ratio (compensates for more raw keypoints)
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

# Fraction of CK output pixels that must be non-zero to accept INSPIRE result.
# Only guard against catastrophic blank output (field folded on itself).

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


def compute_ncc(fixed_f32: np.ndarray, moving_f32: np.ndarray) -> float:
    """
    Compute Normalized Cross-Correlation between two float32 images.
    Returns a value in [-1, 0] where more negative = better alignment.
    Subsamples at every 4th pixel for speed on large images.
    """
    f = fixed_f32[::4, ::4].ravel().astype(np.float64)
    m = moving_f32[::4, ::4].ravel().astype(np.float64)
    f -= f.mean();  f_std = f.std()
    m -= m.mean();  m_std = m.std()
    if f_std < 1e-6 or m_std < 1e-6:
        return 0.0
    ncc = float(np.mean((f / f_std) * (m / m_std)))
    return -abs(ncc)   # return as negative so more negative = better (consistent with SimpleITK NCC)

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

        # INSPIRE expects float32 images normalised to [0, 1].
        # Raw log-normalised uint8 inputs are in [0, 255] — divide by 255.
        tifffile.imwrite(ref_ck_path, (fixed_log_for_reg.astype(np.float32) / 255.0))
        tifffile.imwrite(flo_ck_path, (moving_log_affine.astype(np.float32) / 255.0))

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

        # Apply the displacement field to every channel of the full 16-bit volume.
        # InspireTransform scales output intensity relative to the ref image range.
        # Since ref_ck_path is [0,1] float32, input channels must also be [0,1]
        # so the transform is applied without intensity distortion.
        # We rescale back to the original uint16 range after applying.
        aligned_channels = []
        in_channel_path  = os.path.join(temp_dir, "in_channel.tif")
        out_channel_path = os.path.join(temp_dir, "out_channel.tif")

        for i in range(c):
            ch_f32    = moving_affine_vol[i].astype(np.float32)
            ch_max    = float(ch_f32.max())
            ch_scale  = ch_max if ch_max > 0 else 1.0
            tifffile.imwrite(in_channel_path, (ch_f32 / ch_scale))   # [0, 1]
            trans_cmd = [
                INSPIRE_TRANSFORM_BIN, "-dim", "2",
                "-interpolation", "linear", "-transform", tforward,
                "-ref", ref_ck_path,
                "-in", in_channel_path,
                "-out", out_channel_path,
            ]
            try:
                subprocess.run(trans_cmd, check=True, capture_output=True)
                # Output is float32 in [0,1] — rescale back to original uint16 range
                aligned_f32 = tifffile.imread(out_channel_path).astype(np.float32)
                aligned_ch  = np.clip(aligned_f32 * ch_scale, 0, 65535).astype(np.uint16)
                aligned_channels.append(aligned_ch)
            except subprocess.CalledProcessError as e:
                logger.error(f"[{slice_id}] InspireTransform failed on ch {i}: {e.stderr}")
                return None, False

        aligned_vol = np.stack(aligned_channels, axis=0)  # already uint16 per channel
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

    # Pre-align every channel with the affine transform
    c = moving_np.shape[0]
    moving_affine_vol = np.zeros_like(moving_np, dtype=np.float32)
    for ch in range(c):
        moving_affine_vol[ch] = cv2.warpAffine(
            moving_np[ch].astype(np.float32), M_affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    # L1: INSPIRE elastic — pass affine-prealigned images.
    # INSPIRE always runs its own internal affine stage unconditionally, but
    # since the input is already affine-aligned by AKAZE, its internal affine
    # will find a near-identity transform and proceed directly to the deformable
    # stage which then corrects the residual elastic deformation.
    aligned_np = None
    inspire_ok = False
    moving_log_affine = cv2.warpAffine(
        moving_log, M_affine, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    if akaze_ok:
        aligned_np, inspire_ok = apply_inspire_l1(
            fixed_log, moving_log_affine, moving_affine_vol, sid,
        )

    # Measure NCC on affine-prealigned images as baseline for acceptance.
    # NCC is measured on log-normalised CK (scale-invariant, same as B-spline scripts).
    # NCC is negative — more negative = better alignment.
    if akaze_ok:
        ncc_affine = compute_ncc(
            fixed_log.astype(np.float32),
            moving_log_affine.astype(np.float32),
        )
        logger.info(f"[{sid}] Affine NCC baseline: {ncc_affine:.4f}")
    else:
        ncc_affine = 0.0

    # Output sanity & fallback
    ncc_inspire = 0.0
    if inspire_ok:
        ck_out = aligned_np[CK_CHANNEL_IDX]

        # Check 1: blank output (field folded on itself)
        if np.count_nonzero(ck_out) / float(ck_out.size) < MIN_CK_NONZERO_FRAC:
            logger.warning(
                f"[{sid}] CK output nearly blank — INSPIRE diverged. "
                "Reverting to affine-only result."
            )
            inspire_ok = False
            aligned_np = moving_affine_vol.astype(np.uint16)
        else:
            # Compute NCC for logging/plotting only — not used for acceptance.
            # INSPIRE uses alpha-AMD metric internally which does not correlate
            # with NCC; trusting INSPIRE unless output is blank.
            _, inspire_log = prepare_ck(ck_out.astype(np.float32))
            ncc_inspire    = compute_ncc(
                fixed_log.astype(np.float32),
                inspire_log.astype(np.float32),
            )
            if abs(ncc_affine) > 1e-9:
                relative_improvement = (ncc_affine - ncc_inspire) / abs(ncc_affine)
            else:
                relative_improvement = 0.0
            logger.info(
                f"[{sid}] INSPIRE accepted: NCC affine={ncc_affine:.4f} → "
                f"inspire={ncc_inspire:.4f} ({relative_improvement*100:+.2f}% NCC)"
            )
    else:
        aligned_np = moving_affine_vol.astype(np.uint16)

    # INSPIRE interim plot — always saved when L0 succeeded
    if akaze_ok:
        try:
            save_inspire_plot(
                fixed_log, moving_log_affine,
                aligned_np[CK_CHANNEL_IDX],
                ncc_affine, ncc_inspire, inspire_ok, sid, OUTPUT_FOLDER,
            )
        except Exception as exc:
            logger.warning(f"[{sid}] INSPIRE plot failed: {exc}")

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
        ncc_affine   = round(float(ncc_affine), 6),
        ncc_value    = round(float(ncc_inspire), 6),
    )

    return aligned_np, mse, time.time() - start, stats, akaze_ok, M_affine

# --- PLOTTING ---
def save_inspire_plot(
    fixed_log: np.ndarray,
    moving_log_affine: np.ndarray,
    aligned_ck: np.ndarray,
    ncc_affine: float,
    ncc_inspire: float,
    inspire_ok: bool,
    slice_id: str,
    output_folder: str,
):
    """
    4-panel interim plot for the INSPIRE elastic stage:
      Panel 1 (R/G): Fixed vs Affine-only       — residual before INSPIRE
      Panel 2 (R/G): Fixed vs INSPIRE result     — residual after INSPIRE
      Panel 3:       |Affine CK − INSPIRE CK|    — what elastic correction changed
      Panel 4:       INSPIRE CK greyscale        — sanity check for blank/corrupt output
    """
    out_dir = os.path.join(output_folder, "interim_plots")
    os.makedirs(out_dir, exist_ok=True)

    def norm(x, ref=None):
        # Use ref image's percentile if provided so both images share the same scale
        src = ref if ref is not None else x
        p = np.percentile(src[src > 0], 99.5) if np.any(src > 0) else 1.0
        return np.clip(x.astype(np.float32) / (p if p > 0 else 1), 0, 1)

    f = norm(fixed_log.astype(np.float32))
    a = norm(moving_log_affine.astype(np.float32), ref=fixed_log.astype(np.float32))

    _, aligned_log = prepare_ck(aligned_ck.astype(np.float32))
    b = norm(aligned_log.astype(np.float32), ref=fixed_log.astype(np.float32))

    diff = np.abs(a - b)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(np.dstack((f, a, np.zeros_like(f))))
    axes[0].set_title(f"Fixed (R) vs Affine (G)\nNCC={ncc_affine:.4f}", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(np.dstack((f, b, np.zeros_like(f))))
    axes[1].set_title(f"Fixed (R) vs INSPIRE (G)\nNCC={ncc_inspire:.4f}", fontsize=10)
    axes[1].axis('off')

    im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=diff.max() if diff.max() > 0 else 1)
    axes[2].set_title("|Affine − INSPIRE|\n(what elastic correction changed)", fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.03, pad=0.02)

    axes[3].imshow(b, cmap='gray')
    axes[3].set_title("INSPIRE CK output\n(sanity — should not be blank)", fontsize=10)
    axes[3].axis('off')

    if abs(ncc_affine) > 1e-9 and ncc_inspire != 0.0:
        rel = (ncc_affine - ncc_inspire) / abs(ncc_affine) * 100
        ncc_str = f"  ΔNCC={rel:+.2f}%"
    else:
        ncc_str = ""

    status = "ACCEPTED" if inspire_ok else "BLANK→REVERTED"
    fig.suptitle(
        f"{slice_id}  [{status}]{ncc_str}",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{slice_id}_inspire.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"[{slice_id}] INSPIRE plot saved ({status}).")

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
        f"Min inliers={MIN_INLIERS} | RANSAC thresh={RANSAC_THRESH}px"
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

    # Collect slice IDs only — load slices on demand to avoid OOM.
    # Preloading all slices simultaneously uses ~23GB for a 20-slice core.
    slice_ids = [get_slice_number(f) for f in file_list]

    def load_slice(idx):
        arr = tifffile.imread(file_list[idx])
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)
        if arr.shape[1] != target_h or arr.shape[2] != target_w:
            logger.warning(f"Shape mismatch in {os.path.basename(file_list[idx])} — conforming.")
            arr = conform_slice(arr, target_h, target_w)
        return arr

    aligned_vol             = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    center_raw              = load_slice(center_idx)
    aligned_vol[center_idx] = center_raw
    del center_raw
    anchor_id               = get_slice_number(file_list[center_idx])
    logger.info(f"Volume anchored at slice index {center_idx} (ID {anchor_id})")

    registration_stats = []

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} pass.")

        for i in indices:
            real_id   = get_slice_number(file_list[i])
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = load_slice(i)
            sid       = f"Z{i:03d}_ID{real_id:03d}"

            aligned_np, mse, runtime, stats, success, M_final = register_slice(
                fixed_np, moving_np, slice_id=sid,
            )

            if success and transform_is_sane(M_final):
                aligned_vol[i] = aligned_np
                status_str     = "SUCCESS"
            else:
                aligned_vol[i] = moving_np
                status_str     = "IDENTITY_FALLBACK_RAW"
                logger.warning(f"Z{i:02d} (ID {real_id:03d}): {status_str} — writing raw slice.")

            del moving_np

            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | Det: {stats['detector']} | "
                f"Matches: {stats['n_matches']} | Inliers: {stats['n_inliers']} | "
                f"INSPIRE: {stats['bspline_ok']} | NCC: {stats['ncc_value']:.4f} | MSE: {mse:.2f} | "
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
                "NCC_Value":    stats["ncc_value"],
                "NCC_Affine":   stats["ncc_affine"],
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
        "N_Matches", "N_Inliers", "BSpline_OK", "NCC_Value", "NCC_Affine",
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

    # --- WRITE REGISTERED VOLUME FIRST ---
    # Write before montages so the result is saved even if montage generation fails.
    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
    logger.info(f"Writing registered volume to {out_tiff}")
    try:
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
        logger.info("Registered volume written.")
    except Exception as exc:
        logger.error(f"Volume write failed: {exc}")

    # --- QC MONTAGE ---
    logger.info("Generating QC montage...")
    try:
        generate_qc_montage(
            aligned_vol, OUTPUT_FOLDER,
            slice_ids=slice_ids,
            channel_idx=CK_CHANNEL_IDX,
            channel_name="CK",
            title_suffix="AKAZE+INSPIRE",
        )
    except Exception as exc:
        logger.error(f"Montage failed: {exc}")
    del aligned_vol
    logger.info("Done.")

if __name__ == "__main__":
    main()