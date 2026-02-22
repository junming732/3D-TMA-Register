"""
feature_akaze_rigid.py
======================
Feature registration continuing from Amoeba intensity registered volumes.

Tradeoff: Replaced BRISK descriptor cascade with AKAZE descriptor extraction to optimize for deterministic structural convergence.
Tradeoff: Replaced raw array MSE evaluation with normalized structural domain evaluation to optimize for notebook parity.
Tradeoff: Replaced standard output telemetry with logging to optimize for production auditability.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import tifffile
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
parser = argparse.ArgumentParser(description='Feature registration utilizing AKAZE continuing from Amoeba volumes.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

AMOEBA_OUTPUT     = os.path.join(config.DATASPACE, "Amoeba_Registered_rigid_geometry")
INPUT_VOLUME_PATH = os.path.join(AMOEBA_OUTPUT, TARGET_CORE, f"{TARGET_CORE}_Amoeba_Aligned.ome.tif")
WORK_OUTPUT       = os.path.join(config.DATASPACE, "Feature_Registered_rigid_geometry")
OUTPUT_FOLDER     = os.path.join(WORK_OUTPUT, TARGET_CORE)

CK_CHANNEL_IDX   = 6
MIN_GOOD_MATCHES = 10
LOWE_RATIO       = 0.7 
RANSAC_THRESH    = 3.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def prepare_for_features(img_arr: np.ndarray) -> np.ndarray:
    """
    Normalizes dynamic range and applies CLAHE to amplify structural textures.
    """
    norm  = cv2.normalize(img_arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(norm)

def detect_and_estimate(fixed_8bit: np.ndarray, moving_8bit: np.ndarray):
    """
    Calculates the spatial transformation matrix using AKAZE descriptors exclusively.
    """
    detector = cv2.AKAZE_create()
    norm_type = cv2.NORM_HAMMING
    name = "AKAZE"
    
    kp1, des1 = detector.detectAndCompute(fixed_8bit,  None)
    kp2, des2 = detector.detectAndCompute(moving_8bit, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        logger.warning(f"[{name}] Feature starvation.")
        return None, None, name, 0, 0.0

    matcher     = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good        = [m for m, n in raw_matches if m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_GOOD_MATCHES:
        logger.warning(f"[{name}] Insufficient statistical matches.")
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
    Executes registration and normalizes telemetry to the 8 bit structural domain.
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
        stats = dict(detector="Amoeba_Input", n_good=n_good,
                     inlier_ratio=0.0,
                     rotation_deg=0.0, tx=0.0, ty=0.0, success=False)
        return aligned_np, mse, elapsed, stats, False

    U, S, Vt = np.linalg.svd(M[:2, :2])
    R = U @ Vt
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

    aligned_np = np.stack(aligned_channels, axis=0).astype(np.uint16)
    
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

    plt.suptitle(f'Registration QC AKAZE: {TARGET_CORE}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{TARGET_CORE}_QC_Montage_{channel_name}_Feature.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Montage saved to {out_path}")

def main():
    logger.info(f"Feature Registration Protocol Initiated for {TARGET_CORE}")
    
    if not os.path.exists(INPUT_VOLUME_PATH):
        logger.error(f"Input volume unreadable at {INPUT_VOLUME_PATH}")
        sys.exit(1)

    logger.info("Loading memory volume into tensor array.")
    vol_in = tifffile.imread(INPUT_VOLUME_PATH)

    if vol_in.ndim != 4:
        logger.error(f"Dimensionality constraint failed. Expected 4, detected {vol_in.ndim}.")
        sys.exit(1)

    n_slices = vol_in.shape[0]

    if n_slices < 2:
        logger.warning("Insufficient slice depth. Writing identity output.")
        out_path = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
        tifffile.imwrite(out_path, vol_in, photometric='minisblack',
                         metadata={'axes': 'ZCYX'}, compression='zlib')
        sys.exit(0)

    aligned_vol        = np.zeros_like(vol_in)
    registration_stats = []

    center_idx = n_slices // 2
    aligned_vol[center_idx] = vol_in[center_idx]
    logger.info(f"Volume anchored at slice index {center_idx}")

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} spatial pass.")

        for i in indices:
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = vol_in[i]

            aligned_np, mse, runtime, stats, success = register_slice_feature(fixed_np, moving_np)
            aligned_vol[i] = aligned_np

            status_str = "SUCCESS" if success else "AMOEBA_FALLBACK"
            logger.info(
                f"Z{i:02d} | Protocol: {stats['detector']} | Matches: {stats['n_good']} | "
                f"Inliers: {stats['inlier_ratio']:.2f} | MSE: {mse:.2f} | "
                f"Rot: {stats['rotation_deg']:.2f} | t: {runtime:.2f}s | Status: {status_str}"
            )

            registration_stats.append({
                "Direction":    direction,
                "Slice_Z":      i,
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
    cols = ["Direction", "Slice_Z", "Detector", "N_Matches", "Inlier_Ratio",
            "Success", "Rotation_Deg", "Shift_X_px", "Shift_Y_px", "MSE_After", "Runtime_s"]
    df[cols].to_csv(os.path.join(OUTPUT_FOLDER, "registration_stats_feature.csv"), index=False)

    n_ok       = int(df["Success"].sum())
    n_fallback = int((~df["Success"]).sum())
    logger.info(f"Execution complete. Converged: {n_ok} | Amoeba Fallbacks: {n_fallback}")

    generate_qc_montage(aligned_vol, OUTPUT_FOLDER, channel_idx=CK_CHANNEL_IDX, channel_name="CK")

    out_tiff = os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_Feature_Aligned.ome.tif")
    logger.info(f"Committing registered tensor to disk at {out_tiff}")
    tifffile.imwrite(out_tiff, aligned_vol, photometric='minisblack',
                     metadata={'axes': 'ZCYX'}, compression='zlib')

if __name__ == "__main__":
    main()