"""
Multi-Detector Feature Extraction Sandbox.
Evaluates AKAZE, BRISK, ORB, and SIFT across multiple preprocessing pipelines.
Injects initial downsampling for rapid iterative exploration.
Outputs raw metrics to CSV and side-by-side visual diagnostics.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import tifffile
import glob
import re
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description='Evaluate multi-detector feature pipelines.')
parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
args = parser.parse_args()

TARGET_CORE = args.core_name

# Dynamic paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config 

DATA_BASE_PATH  = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate")
INPUT_FOLDER    = os.path.join(DATA_BASE_PATH, TARGET_CORE)
WORK_OUTPUT     = os.path.join(config.DATASPACE, "Exploration_Multi")
OUTPUT_FOLDER   = os.path.join(WORK_OUTPUT, TARGET_CORE)

CK_CHANNEL_IDX   = 6
MIN_GOOD_MATCHES = 5
LOWE_RATIO       = 0.75
RANSAC_THRESH    = 3.0
ANMS_KEEP_POINTS = 1000
DOWNSAMPLE_SCALE = 0.5  # 50% scale for rapid exploration

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --- 1. DOWNSAMPLING & PREPROCESSING SUITE ---

def prep_linear_stretch(img_arr: np.ndarray) -> np.ndarray:
    p_low, p_high = np.percentile(img_arr[::4, ::4], (0.1, 99.9))
    clipped = np.clip(img_arr, p_low, p_high)
    return cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def prep_clahe(img_arr: np.ndarray) -> np.ndarray:
    norm = prep_linear_stretch(img_arr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(norm)

def prep_log_transform(img_arr: np.ndarray) -> np.ndarray:
    img_float = img_arr.astype(np.float32)
    log_img = np.log1p(img_float)
    p_low, p_high = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    clipped = np.clip(log_img, p_low, p_high)
    return cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def prep_gamma_correction(img_arr: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    norm = prep_linear_stretch(img_arr).astype(np.float32) / 255.0
    gamma_corrected = np.power(norm, gamma) * 255.0
    return gamma_corrected.astype(np.uint8)

PREPROCESSORS = {
    "01_Linear": prep_linear_stretch,
    "02_CLAHE": prep_clahe,
    "03_Log": prep_log_transform,
    "04_Gamma0.6": lambda x: prep_gamma_correction(x, gamma=0.6)
}

# Define detectors and their strictly required norms
DETECTORS = {
    "AKAZE": (cv2.AKAZE_create(), cv2.NORM_HAMMING),
    "BRISK": (cv2.BRISK_create(), cv2.NORM_HAMMING),
    "ORB":   (cv2.ORB_create(nfeatures=5000), cv2.NORM_HAMMING),
    "SIFT":  (cv2.SIFT_create(), cv2.NORM_L2)
}


# --- 2. DETECTION AND GEOMETRY MODULES ---

def filter_to_strict_rigid(M_partial: np.ndarray) -> np.ndarray:
    if M_partial is None: return None
    M_strict = M_partial.copy()
    R_scale = M_strict[0:2, 0:2]
    scale = np.sqrt(R_scale[0, 0]**2 + R_scale[1, 0]**2)
    if scale > 1e-6:
        M_strict[0:2, 0:2] = R_scale / scale
    return M_strict

def apply_anms_with_descriptors(keypoints: tuple, descriptors: np.ndarray, num_to_keep: int = 1000, c_robust: float = 0.9, max_compute_pts: int = 5000):
    n_kpts = len(keypoints)
    if n_kpts <= num_to_keep or descriptors is None:
        return keypoints, descriptors

    coords = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])

    sort_idx = np.argsort(responses)[::-1]
    if len(sort_idx) > max_compute_pts:
        sort_idx = sort_idx[:max_compute_pts]
        n_kpts = max_compute_pts
        
    coords = coords[sort_idx]
    responses = responses[sort_idx]
    radii = np.full(n_kpts, np.inf)

    for i in range(1, n_kpts):
        stronger_mask = responses[:i] * c_robust > responses[i]
        if np.any(stronger_mask):
            diffs = coords[:i][stronger_mask] - coords[i]
            dists_sq = np.sum(diffs**2, axis=1)
            radii[i] = np.min(dists_sq)

    best_idx = np.argsort(radii)[::-1][:num_to_keep]
    final_idx = sort_idx[best_idx]
    
    return tuple(keypoints[i] for i in final_idx), descriptors[final_idx]

def detect_and_estimate(fixed_8bit: np.ndarray, moving_8bit: np.ndarray, detector_obj, norm_type):
    kp1, des1 = detector_obj.detectAndCompute(fixed_8bit, None)
    kp2, des2 = detector_obj.detectAndCompute(moving_8bit, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        # CORRECTED: Return None instead of [] so the visualizer safely skips
        return None, None, None, len(kp1), len(kp2)

    kp1_anms, des1_anms = apply_anms_with_descriptors(kp1, des1, num_to_keep=ANMS_KEEP_POINTS)
    kp2_anms, des2_anms = apply_anms_with_descriptors(kp2, des2, num_to_keep=ANMS_KEEP_POINTS)

    matcher = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(des1_anms, des2_anms, k=2)
    good = [m for m, n in raw_matches if m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_GOOD_MATCHES:
        # CORRECTED: Return a proper dictionary. 
        # This allows the visualizer to plot the keypoints even if the matcher fails.
        diagnostics = {
            "kp_fixed_anms": kp1_anms, "kp_moving_anms": kp2_anms,
            "good_matches": good, "inlier_mask": None
        }
        return None, None, diagnostics, len(kp1), len(kp2)

    src_pts = np.float32([kp1_anms[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2_anms[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M_partial, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH)
    M_strict = filter_to_strict_rigid(M_partial)

    diagnostics = {
        "kp_fixed_anms": kp1_anms, "kp_moving_anms": kp2_anms,
        "good_matches": good, "inlier_mask": mask.ravel() if mask is not None else None
    }

    return M_strict, mask, diagnostics, len(kp1), len(kp2)


# --- 3. VISUALIZATION EXPORT MODULE ---

def dump_pair_visuals(method_name: str, detector_name: str, pair_id: str, fixed_img: np.ndarray, moving_img: np.ndarray, diag: dict):
    """
    Saves side-by-side keypoint and match visualizations.
    Utilizes copyMakeBorder to prevent hstack shape mismatch crashes.
    """
    if not diag: return
    
    out_dir = os.path.join(OUTPUT_FOLDER, method_name, detector_name)
    os.makedirs(out_dir, exist_ok=True)
    
    fixed_kp_img = cv2.drawKeypoints(fixed_img, diag["kp_fixed_anms"], None, color=(255, 0, 0))
    moving_kp_img = cv2.drawKeypoints(moving_img, diag["kp_moving_anms"], None, color=(255, 0, 0))
    
    h1 = fixed_kp_img.shape[0]
    h2 = moving_kp_img.shape[0]
    max_h = max(h1, h2)
    
    if h1 < max_h:
        fixed_kp_img = cv2.copyMakeBorder(fixed_kp_img, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    if h2 < max_h:
        moving_kp_img = cv2.copyMakeBorder(moving_kp_img, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        
    kp_concat = np.hstack((fixed_kp_img, moving_kp_img))
    cv2.imwrite(os.path.join(out_dir, f"{pair_id}_01_Keypoints.png"), kp_concat)

    if diag.get("inlier_mask") is not None:
        inliers = [m for i, m in enumerate(diag["good_matches"]) if diag["inlier_mask"][i]]
        match_img = cv2.drawMatches(
            fixed_img, diag["kp_fixed_anms"], 
            moving_img, diag["kp_moving_anms"], 
            inliers, None, 
            matchColor=(0, 0, 255),       
            singlePointColor=(255, 0, 0), 
            flags=2
        )
        cv2.imwrite(os.path.join(out_dir, f"{pair_id}_02_Matches.png"), match_img)


# --- 4. ORCHESTRATION LOOP ---

def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0

def main():
    logger.info(f"Initiating Multi-Detector Exploration for {TARGET_CORE} at {DOWNSAMPLE_SCALE}x scale.")

    raw_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.ome.tif")), key=get_slice_number)
    if len(raw_files) < 2:
        logger.error("Insufficient files for pairwise evaluation.")
        sys.exit(1)

    all_stats = []

    for i in range(len(raw_files) - 1):
        f_fixed, f_moving = raw_files[i], raw_files[i+1]
        pair_id = f"Z{get_slice_number(f_fixed):03d}_vs_Z{get_slice_number(f_moving):03d}"
        logger.info(f"Evaluating Pair: {pair_id}")

        try:
            arr_fixed, arr_moving = tifffile.imread(f_fixed), tifffile.imread(f_moving)
            if arr_fixed.ndim == 3 and arr_fixed.shape[-1] < arr_fixed.shape[0]:
                arr_fixed, arr_moving = np.moveaxis(arr_fixed, -1, 0), np.moveaxis(arr_moving, -1, 0)
                
            ck_fixed_raw = arr_fixed[CK_CHANNEL_IDX].astype(np.float32)
            ck_moving_raw = arr_moving[CK_CHANNEL_IDX].astype(np.float32)
            
            h, w = ck_fixed_raw.shape
            ck_fixed_down = cv2.resize(ck_fixed_raw, (int(w * DOWNSAMPLE_SCALE), int(h * DOWNSAMPLE_SCALE)), interpolation=cv2.INTER_AREA)
            ck_moving_down = cv2.resize(ck_moving_raw, (int(w * DOWNSAMPLE_SCALE), int(h * DOWNSAMPLE_SCALE)), interpolation=cv2.INTER_AREA)

        except Exception as e:
            logger.error(f"Failed to load pair {pair_id}: {e}")
            continue

        for method_name, prep_func in PREPROCESSORS.items():
            ck_fixed_8bit = prep_func(ck_fixed_down)
            ck_moving_8bit = prep_func(ck_moving_down)

            for det_name, (det_obj, norm_type) in DETECTORS.items():
                
                M, mask, diag, kp_f_cnt, kp_m_cnt = detect_and_estimate(ck_fixed_8bit, ck_moving_8bit, det_obj, norm_type)
                n_inliers = int(mask.sum()) if mask is not None else 0
                
                all_stats.append({
                    "Pair": pair_id,
                    "Method": method_name,
                    "Detector": det_name,
                    "RANSAC_Inliers": n_inliers,
                })
                
                dump_pair_visuals(method_name, det_name, pair_id, ck_fixed_8bit, ck_moving_8bit, diag)

    pd.DataFrame(all_stats).to_csv(os.path.join(OUTPUT_FOLDER, f"{TARGET_CORE}_multi_exploration_stats.csv"), index=False)
    logger.info("Exploration complete. Metrics and visuals saved.")

if __name__ == "__main__":
    main()