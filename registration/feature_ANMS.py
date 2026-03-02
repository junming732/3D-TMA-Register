"""
Feature registration as the FIRST stage in the pipeline.
Reads raw .ome.tif slices directly at FULL resolution (no prior intensity registration).

Tradeoff: Operates on full-resolution input — higher fidelity but slower than downsampled approach.
Tradeoff: Uses AKAZE descriptors for deterministic structural convergence.
Tradeoff: Replaced standard output telemetry with logging for production auditability.
Update: Integrated Brown Adaptive Non-Maximal Suppression (ANMS) and interim CLAHE/Keypoint diagnostics.
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
DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate")
INPUT_FOLDER      = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT       = os.path.join(config.DATASPACE, "Feature_ANMS")
OUTPUT_FOLDER     = os.path.join(WORK_OUTPUT, TARGET_CORE)

CK_CHANNEL_IDX   = 6
MIN_GOOD_MATCHES = 10
LOWE_RATIO       = 0.75
RANSAC_THRESH    = 3.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)
ANMS_KEEP_POINTS = 1000

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
    """
    c, h, w = arr.shape
    out = np.zeros((c, target_h, target_w), dtype=arr.dtype)

    if h >= target_h:
        src_y0 = (h - target_h) // 2
        src_y1 = src_y0 + target_h
        dst_y0, dst_y1 = 0, target_h
    else:
        src_y0, src_y1 = 0, h
        dst_y0 = (target_h - h) // 2
        dst_y1 = dst_y0 + h

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


def prepare_for_features_with_diagnostics(img_arr: np.ndarray) -> tuple:
    """
    Normalizes dynamic range using strided percentile clipping for speed.
    """
    # OPTIMIZATION: Calculate percentiles on a 16x downsampled view of the array
    p_low, p_high = np.percentile(img_arr[::4, ::4], (1, 99.9))
    
    clipped = np.clip(img_arr, p_low, p_high)
    norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    clahe_applied = clahe.apply(norm)
    
    return norm, clahe_applied



def save_interim_diagnostics(
    output_dir: str, 
    slice_id: str, 
    img_raw_8bit: np.ndarray, 
    img_clahe_8bit: np.ndarray,
    kp_raw: list, 
    kp_anms: list,
    img_fixed_clahe: np.ndarray = None,
    kp_fixed_anms: list = None,
    good_matches: list = None,
    inlier_mask: np.ndarray = None
):
    """
    Generates and saves interim state visualizations for the registration pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Before vs After CLAHE
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_raw_8bit, cmap='gray')
    axes[0].set_title("1. Original (Normalized)")
    axes[0].axis('off')
    
    axes[1].imshow(img_clahe_8bit, cmap='gray')
    axes[1].set_title("2. After CLAHE")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{slice_id}_01_CLAHE_comparison.png"), dpi=150)
    plt.close(fig)

    # 2. Original Keypoints
    img_kp_raw = cv2.drawKeypoints(img_clahe_8bit, kp_raw, None, color=(0, 255, 0), flags=0)
    cv2.imwrite(os.path.join(output_dir, f"{slice_id}_02_Raw_Keypoints_{len(kp_raw)}.png"), img_kp_raw)

    # 3. ANMS Keypoints
    if kp_anms:
        img_kp_anms = cv2.drawKeypoints(img_clahe_8bit, kp_anms, None, color=(255, 0, 0), flags=0)
        cv2.imwrite(os.path.join(output_dir, f"{slice_id}_03_ANMS_Keypoints_{len(kp_anms)}.png"), img_kp_anms)

    # 4. Final Inliers
    if good_matches and inlier_mask is not None and img_fixed_clahe is not None:
        inlier_matches = [m for i, m in enumerate(good_matches) if inlier_mask[i]]
        img_inliers = cv2.drawMatches(
            img_fixed_clahe, kp_fixed_anms, 
            img_clahe_8bit, kp_anms, 
            inlier_matches, None, 
            matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=2
        )
        cv2.imwrite(os.path.join(output_dir, f"{slice_id}_04_Final_Inliers_{len(inlier_matches)}.png"), img_inliers)


def apply_anms_with_descriptors(keypoints: tuple, descriptors: np.ndarray, num_to_keep: int = 1000, c_robust: float = 0.9, max_compute_pts: int = 5000):
    """
    Applies Brown ANMS to keypoints. 
    Optimized: Truncates the input to `max_compute_pts` before running the O(N^2) radius calculation.
    """
    n_kpts = len(keypoints)
    if n_kpts <= num_to_keep or descriptors is None:
        return keypoints, descriptors

    coords = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])

    # Sort descending by response strength
    sort_idx = np.argsort(responses)[::-1]
    
    # OPTIMIZATION: Truncate to prevent O(N^2) explosion
    if len(sort_idx) > max_compute_pts:
        sort_idx = sort_idx[:max_compute_pts]
        n_kpts = max_compute_pts
        
    coords = coords[sort_idx]
    responses = responses[sort_idx]
    
    radii = np.full(n_kpts, np.inf)

    # O(N^2) loop is now bounded by max_compute_pts
    for i in range(1, n_kpts):
        stronger_mask = responses[:i] * c_robust > responses[i]
        if np.any(stronger_mask):
            diffs = coords[:i][stronger_mask] - coords[i]
            dists_sq = np.sum(diffs**2, axis=1)
            radii[i] = np.min(dists_sq)

    best_idx = np.argsort(radii)[::-1][:num_to_keep]
    final_idx = sort_idx[best_idx]
    
    filtered_kp = tuple(keypoints[i] for i in final_idx)
    filtered_des = descriptors[final_idx]
    
    return filtered_kp, filtered_des

def filter_to_strict_rigid(M_partial: np.ndarray) -> np.ndarray:
    """
    Strips the uniform scale factor from an OpenCV 4-DOF partial affine matrix.
    Enforces a strict 3-DOF Euclidean constraint (Translation + Rotation only).
    """
    if M_partial is None:
        return None
        
    M_strict = M_partial.copy()
    R_scale = M_strict[0:2, 0:2]
    
    # Calculate the isotropic scale factor (magnitude of the first column vector)
    scale = np.sqrt(R_scale[0, 0]**2 + R_scale[1, 0]**2)
    
    # Normalize the rotation block to force scale = 1.0, preventing zero-division
    if scale > 1e-6:
        M_strict[0:2, 0:2] = R_scale / scale
        
    return M_strict

def detect_and_estimate(fixed_8bit: np.ndarray, moving_8bit: np.ndarray, anms_keep: int = ANMS_KEEP_POINTS):
    """
    Calculates the spatial transformation matrix using AKAZE, ANMS, and Rigid (Partial Affine) consensus.
    """
    detector  = cv2.AKAZE_create()
    norm_type = cv2.NORM_HAMMING
    name      = "AKAZE_ANMS_Rigid"

    # 1. Raw Detection
    kp1, des1 = detector.detectAndCompute(fixed_8bit,  None)
    kp2, des2 = detector.detectAndCompute(moving_8bit, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        logger.warning(f"[{name}] Feature starvation at raw detection.")
        return None, None, name, 0, 0.0, None

    # 2. Adaptive Non-Maximal Suppression
    kp1_anms, des1_anms = apply_anms_with_descriptors(kp1, des1, num_to_keep=anms_keep)
    kp2_anms, des2_anms = apply_anms_with_descriptors(kp2, des2, num_to_keep=anms_keep)

    # 3. Descriptor Matching
    matcher     = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(des1_anms, des2_anms, k=2)
    
    # Lowe's Ratio Test
    good = [m for m, n in raw_matches if m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_GOOD_MATCHES:
        logger.warning(f"[{name}] Insufficient statistical matches ({len(good)}).")
        return None, None, name, len(good), 0.0, None

    # 4. Geometric Consensus (Rigid / Partial Affine)
    src_pts = np.float32([kp1_anms[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2_anms[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M_partial, mask = cv2.estimateAffinePartial2D(
        dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH
    )

    if M_partial is None or mask is None:
        logger.warning(f"[{name}] Rigid estimation divergence.")
        return None, None, name, len(good), 0.0, None

    # 5. Enforce Strict 3-DOF Euclidean Transform
    M_strict = filter_to_strict_rigid(M_partial)

    inlier_ratio = float(mask.sum()) / len(mask)
    
    diagnostics = {
        "kp_fixed_raw": kp1,
        "kp_moving_raw": kp2,
        "kp_fixed_anms": kp1_anms,
        "kp_moving_anms": kp2_anms,
        "good_matches": good,
        "inlier_mask": mask.ravel()
    }

    # Return the strictly rigid matrix
    return M_strict, mask, name, len(good), inlier_ratio, diagnostics


def register_slice_feature(fixed_np: np.ndarray, moving_np: np.ndarray, slice_id: str = None, diag_dir: str = None):
    """
    Executes feature registration at full resolution.
    Routes ANMS interim states to the diagnostic visualization module.
    """
    start = time.time()

    fixed_ck  = fixed_np[CK_CHANNEL_IDX].astype(np.float32)
    moving_ck = moving_np[CK_CHANNEL_IDX].astype(np.float32)

    fixed_norm, fixed_8bit   = prepare_for_features_with_diagnostics(fixed_ck)
    moving_norm, moving_8bit = prepare_for_features_with_diagnostics(moving_ck)

    h, w = fixed_8bit.shape

    M, mask, detector, n_good, inlier_ratio, diagnostics = detect_and_estimate(fixed_8bit, moving_8bit)

    if diagnostics is not None and slice_id is not None and diag_dir is not None:
        save_interim_diagnostics(
            output_dir=diag_dir,
            slice_id=slice_id,
            img_raw_8bit=moving_norm,
            img_clahe_8bit=moving_8bit,
            kp_raw=diagnostics["kp_moving_raw"],
            kp_anms=diagnostics["kp_moving_anms"],
            img_fixed_clahe=fixed_8bit,
            kp_fixed_anms=diagnostics["kp_fixed_anms"],
            good_matches=diagnostics["good_matches"],
            inlier_mask=diagnostics["inlier_mask"]
        )

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
    _, warped_8bit = prepare_for_features_with_diagnostics(aligned_np[CK_CHANNEL_IDX].astype(np.float32))
    
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


def generate_qc_montage(vol: np.ndarray, output_folder: str, channel_idx: int = 6, channel_name: str = "CK"):
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

    center_idx = n_slices // 2
    _center_arr = tifffile.imread(file_list[center_idx])
    if _center_arr.ndim == 2:
        _center_arr = _center_arr[np.newaxis]
    elif _center_arr.ndim == 3 and _center_arr.shape[-1] < _center_arr.shape[0]:
        _center_arr = np.moveaxis(_center_arr, -1, 0)
    c, target_h, target_w = _center_arr.shape
    logger.info(f"Canonical shape set from center slice: C={c}, H={target_h}, W={target_w}")

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

    aligned_vol = np.zeros((n_slices, c, target_h, target_w), dtype=np.uint16)
    registration_stats = []

    aligned_vol[center_idx] = raw_slices[center_idx]
    anchor_id = get_slice_number(file_list[center_idx])
    logger.info(f"Volume anchored at slice index {center_idx} (ID {anchor_id})")

    def process_pass(indices, direction):
        fixed_offset = 1 if direction == "Backward" else -1
        logger.info(f"Executing {direction} spatial pass.")
        
        diag_out_dir = os.path.join(OUTPUT_FOLDER, "Diagnostics_ANMS")

        for i in indices:
            real_id   = get_slice_number(file_list[i])
            fixed_np  = aligned_vol[i + fixed_offset]
            moving_np = raw_slices[i]

            aligned_np, mse, runtime, stats, success = register_slice_feature(
                fixed_np, 
                moving_np,
                slice_id=f"Z{i:03d}_ID{real_id:03d}",
                diag_dir=diag_out_dir
            )
            aligned_vol[i] = aligned_np

            status_str = "SUCCESS" if success else "IDENTITY_FALLBACK"
            status_str = "SUCCESS" if success else "IDENTITY_FALLBACK"
            logger.info(
                f"Z{i:02d} (ID {real_id:03d}) | Det: {stats['detector']} | "
                f"Match: {stats['n_good']} | Inliers: {stats['inlier_ratio']:.2f} | "
                f"MSE: {mse:.2f} | Rot: {stats['rotation_deg']:.2f}° | "
                f"tx: {stats['tx']:.1f}px | ty: {stats['ty']:.1f}px | "
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