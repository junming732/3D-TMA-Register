"""
Feature registration as the FIRST stage in the pipeline.
Reads raw .ome.tif slices directly at FULL resolution (no prior intensity registration).

Tradeoff: Operates on full-resolution input — higher fidelity but slower than downsampled approach.
Tradeoff: Uses AKAZE descriptors for deterministic structural convergence.
Tradeoff: Replaced standard output telemetry with logging for production auditability.
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
import matplotlib
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURATION AND TUNING PARAMETERS
parser = argparse.ArgumentParser(description='Feature registration (AKAZE) — first pipeline stage, full resolution.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# --- PATHS ---
# Input: raw .ome.tif slices (no prior registration)
DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_NEW")
INPUT_FOLDER      = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT       = os.path.join(config.DATASPACE, "Feature_Registered_AKAZE_First")
OUTPUT_FOLDER     = os.path.join(WORK_OUTPUT, TARGET_CORE)

CK_CHANNEL_IDX   = 6
MIN_GOOD_MATCHES = 10
LOWE_RATIO       = 0.7
RANSAC_THRESH    = 3.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)

if not os.path.exists(INPUT_FOLDER):
    logger.error(f"Input folder not found: {INPUT_FOLDER}")
    sys.exit(1)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def conform_slice(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Pads or centre-crops a (C, H, W) slice to (C, target_h, target_w).
    Slices that are slightly smaller are zero-padded symmetrically;
    slices that are slightly larger are centre-cropped.
    This handles the common case of minor sensor read-out size differences
    across sections (e.g. 6048 vs 6080).
    """
    c, h, w = arr.shape
    out = np.zeros((c, target_h, target_w), dtype=arr.dtype)

    # Compute crop/paste windows for H
    if h >= target_h:
        src_y0 = (h - target_h) // 2
        src_y1 = src_y0 + target_h
        dst_y0, dst_y1 = 0, target_h
    else:
        src_y0, src_y1 = 0, h
        dst_y0 = (target_h - h) // 2
        dst_y1 = dst_y0 + h

    # Compute crop/paste windows for W
    if w >= target_w:
        src_x0 = (w - target_w) // 2
        src_x1 = src_x0 + target_w
        dst_x0, dst_x1 = 0, target_w
    else:
        src_x0, src_x1 = 0, w
        dst_x0 = (target_w - w) // 2
        dst_x1 = dst_x0 + w

    out[:, dst_y0:dst_y1, dst_x0:dst_x1] = arr[:, src_y0:src_y1, src_x0:src_x1]
    return out


def prepare_for_features(img_arr: np.ndarray) -> np.ndarray:
    """
    Normalizes dynamic range and applies CLAHE to amplify structural textures.
    """
    norm  = cv2.normalize(img_arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(norm)


def detect_and_estimate(fixed_8bit: np.ndarray, moving_8bit: np.ndarray):
    """
    Calculates the spatial transformation matrix using AKAZE descriptors.
    """
    detector  = cv2.AKAZE_create()
    norm_type = cv2.NORM_HAMMING
    name      = "AKAZE"

    kp1, des1 = detector.detectAndCompute(fixed_8bit,  None)
    kp2, des2 = detector.detectAndCompute(moving_8bit, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        logger.warning(f"[{name}] Feature starvation.")
        return None, None, name, 0, 0.0

    matcher     = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good        = [m for m, n in raw_matches if m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_GOOD_MATCHES:
        logger.warning(f"[{name}] Insufficient statistical matches ({len(good)}).")
        return None, None, name, len(good), 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC,
                                    ransacReprojThreshold=RANSAC_THRESH)

    if M is None or mask is None:
        logger.warning(f"[{name}] Affine estimation divergence.")
        return None, None, name, len(good), 0.0

    inlier_ratio = float(mask.sum()) / len(mask)
    return M, mask, name, len(good), inlier_ratio


def register_slice_feature(fixed_np: np.ndarray, moving_np: np.ndarray):
    """
    Executes feature registration at full resolution.
    fixed_np / moving_np: shape (C, H, W), dtype uint16.
    """
    start = time.time()

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_8bit  = prepare_for_features(fixed_ck)
    moving_8bit = prepare_for_features(moving_ck)

    h, w = fixed_8bit.shape

    M, mask, detector, n_good, inlier_ratio = detect_and_estimate(fixed_8bit, moving_8bit)

    if M is None:
        aligned_np = moving_np.copy()
        mse = float(np.mean((fixed_8bit.astype(np.float32) - moving_8bit.astype(np.float32)) ** 2))
        elapsed = time.time() - start
        stats = dict(detector="Identity_Fallback", n_good=n_good,
                     inlier_ratio=0.0, rotation_deg=0.0, tx=0.0, ty=0.0, success=False)
        return aligned_np, mse, elapsed, stats, False

    U, S, Vt = np.linalg.svd(M[:2, :2])
    R   = U @ Vt
    rot = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    tx  = float(M[0, 2])
    ty  = float(M[1, 2])

    aligned_channels = []
    for c in range(fixed_np.shape[0]):
        ch     = moving_np[c].astype(np.float32)
        warped = cv2.warpAffine(ch, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        aligned_channels.append(warped)

    aligned_np  = np.stack(aligned_channels, axis=0).astype(np.uint16)
    warped_8bit = prepare_for_features(aligned_np[CK_CHANNEL_IDX].astype(np.float32))
    mse         = float(np.mean((fixed_8bit.astype(np.float32) - warped_8bit.astype(np.float32)) ** 2))
    elapsed     = time.time() - start

    stats = dict(
        detector     = detector,
        n_good       = n_good,
        inlier_ratio = round(inlier_ratio, 3),
        rotation_deg = round(rot, 3),
        tx           = round(tx, 3),
        ty           = round(ty, 3),
        success      = True
    )
    return aligned_np, mse, elapsed, stats, True


def generate_qc_montage(vol: np.ndarray, output_folder: str,
                         channel_idx: int = 6, channel_name: str = "CK"):
    logger.info(f"Generating QC montage for {channel_name} channel.")
    n_slices = vol.shape[0]
    if n_slices < 2:
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
        axes_flat[idx].set_title(f"Z{z1} to Z{z2}", fontsize=10, fontweight='bold')
        axes_flat[idx].axis('off')

    for idx in range(len(all_pairs), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.suptitle(f'Registration QC AKAZE (Stage 1): {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{TARGET_CORE}_QC_Montage_{channel_name}_Feature.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")


def main():
    logger.info(f"Feature Registration (Stage 1 / Full Resolution) Initiated for {TARGET_CORE}")

    raw_files  = glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif"))
    file_list  = sorted(raw_files, key=get_slice_number)
    n_slices   = len(file_list)

    if n_slices == 0:
        logger.error(f"No .ome.tif files found in {INPUT_FOLDER}")
        sys.exit(1)

    if n_slices < 2:
        logger.warning("Insufficient slice depth. Writing identity output.")
        vol_in   = tifffile.imread(file_list[0])
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
        tifffile.imwrite(out_path, vol_in[np.newaxis], photometric='minisblack',
                         metadata={'axes': 'ZCYX'}, compression='zlib')
        sys.exit(0)

    # Determine canonical shape from the center slice
    center_idx = n_slices // 2
    _center_arr = tifffile.imread(file_list[center_idx])
    if _center_arr.ndim == 2:
        _center_arr = _center_arr[np.newaxis]
    elif _center_arr.ndim == 3 and _center_arr.shape[-1] < _center_arr.shape[0]:
        _center_arr = np.moveaxis(_center_arr, -1, 0)
    c, target_h, target_w = _center_arr.shape
    logger.info(f"Canonical shape set from center slice: C={c}, H={target_h}, W={target_w}")

    # Load all slices into memory as numpy arrays (C, H, W), conforming to canonical shape
    logger.info(f"Loading {n_slices} slices into memory.")
    raw_slices = []
    for f in file_list:
        arr = tifffile.imread(f)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)
        if arr.shape[1] != target_h or arr.shape[2] != target_w:
            logger.warning(f"Shape mismatch {arr.shape[1:]} vs canonical ({target_h},{target_w}) "
                           f"in {os.path.basename(f)} — conforming via pad/crop.")
            arr = conform_slice(arr, target_h, target_w)
        raw_slices.append(arr)

    # Pre-allocate aligned volume using canonical shape
    aligned_vol = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)

    registration_stats = []

    # Anchor: copy raw center slice directly (already conformed)
    aligned_vol[center_idx] = raw_slices[center_idx]
    anchor_id = get_slice_number(file_list[center_idx])
    logger.info(f"Volume anchored at slice index {center_idx} (ID {anchor_id})")

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} spatial pass.")

        for i in indices:
            real_id   = get_slice_number(file_list[i])
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = raw_slices[i]

            aligned_np, mse, runtime, stats, success = register_slice_feature(fixed_np, moving_np)
            aligned_vol[i] = aligned_np

            status_str = "SUCCESS" if success else "IDENTITY_FALLBACK"
            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | Detector: {stats['detector']} | "
                f"Matches: {stats['n_good']} | Inliers: {stats['inlier_ratio']:.2f} | "
                f"MSE: {mse:.2f} | Rot: {stats['rotation_deg']:.2f}° | "
                f"t: {runtime:.2f}s | Status: {status_str}"
            )

            registration_stats.append({
                "Direction":    direction,
                "Slice_Z":      i,
                "Slice_ID":     real_id,
                "Detector":     stats["detector"],
                "N_Matches":    stats["n_good"],
                "Inlier_Ratio": stats["inlier_ratio"],
                "Success":      success,
                "MSE_After":    round(mse, 4),
                "Rotation_Deg": stats["rotation_deg"],
                "Shift_X_px":   stats["tx"],
                "Shift_Y_px":   stats["ty"],
                "Runtime_s":    round(runtime, 3),
            })

    if center_idx > 0:
        process_pass(range(center_idx - 1, -1, -1), "Backward")
    if center_idx < n_slices - 1:
        process_pass(range(center_idx + 1, n_slices), "Forward")

    df   = pd.DataFrame(registration_stats).sort_values("Slice_Z")
    cols = ["Direction", "Slice_Z", "Slice_ID", "Detector", "N_Matches", "Inlier_Ratio",
            "Success", "Rotation_Deg", "Shift_X_px", "Shift_Y_px", "MSE_After", "Runtime_s"]
    df[cols].to_csv(os.path.join(OUTPUT_FOLDER, "registration_stats_feature.csv"), index=False)

    n_ok       = int(df["Success"].sum())
    n_fallback = int((~df["Success"]).sum())
    logger.info(f"Execution complete. Converged: {n_ok} | Identity Fallbacks: {n_fallback}")

    generate_qc_montage(aligned_vol, OUTPUT_FOLDER, channel_idx=CK_CHANNEL_IDX, channel_name="CK")

    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
    logger.info(f"Committing registered tensor to disk at {out_tiff}")
    tifffile.imwrite(out_tiff, aligned_vol, photometric='minisblack',
                     metadata={'axes': 'ZCYX'}, compression='zlib')
    logger.info("Done.")


if __name__ == "__main__":
    main()