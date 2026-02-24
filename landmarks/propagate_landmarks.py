import os
import sys
import cv2
import numpy as np
import tifffile
import pandas as pd
import json
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config
    INPUT_ROOT = os.path.join(config.DATASPACE, "downsampled_4x")
    LANDMARK_FILE = os.path.join(config.DATASPACE, "manual_landmarks.csv")
except ImportError:
    INPUT_ROOT = r"./downsampled_4x"
    LANDMARK_FILE = r"./manual_landmarks.csv"

OUTPUT_JSON = os.path.join(INPUT_ROOT, "propagated_landmarks.json")
CK_INDEX = 6 

# --- GUARDED PARAMETERS ---
TEMPLATE_SIZE = 61       # 61x61 (Approx 120um). Large enough for context, small enough to avoid too much background.
SEARCH_RESTRICTION = 40  # Pixels. The point cannot drift more than 40px from the "Global Shift" prediction.
MIN_INTENSITY = 25       # "The Void Guard". Any pixel darker than 25/255 is invalid background.

def load_image_ck(filepath):
    try:
        img = tifffile.imread(filepath)
        if img.shape[0] < 20: ck = img[CK_INDEX, :, :]
        else: ck = img[:, :, CK_INDEX]
        # Normalize
        return cv2.normalize(ck, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    except: return None

def pad_to_match(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    max_h, max_w = max(h1, h2), max(w1, w2)
    if h1 < max_h or w1 < max_w:
        img1 = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, max_w - w1, cv2.BORDER_CONSTANT, value=0)
    if h2 < max_h or w2 < max_w:
        img2 = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, max_w - w2, cv2.BORDER_CONSTANT, value=0)
    return img1, img2

def get_global_shift(img_prev, img_next):
    """
    Calculates the 'Average' movement of the whole tissue.
    This acts as a 'Leash' to prevent individual points from wandering off.
    """
    # Downscale for speed/robustness
    curr_small = cv2.resize(img_prev, None, fx=0.25, fy=0.25)
    next_small = cv2.resize(img_next, None, fx=0.25, fy=0.25)
    
    # Phase Correlation
    hann = cv2.createHanningWindow((curr_small.shape[1], curr_small.shape[0]), cv2.CV_32F)
    shift, _ = cv2.phaseCorrelate(np.float32(curr_small), np.float32(next_small), window=hann)
    
    return shift[0] * 4.0, shift[1] * 4.0

def track_guarded(img_anchor, img_target, p_anchor, global_guess):
    """
    Matches the Anchor Template, but restricted to the 'Global Guess' area.
    Includes a 'Void Guard' to reject black pixels.
    """
    x_anc, y_anc = p_anchor.ravel()
    h, w = img_anchor.shape
    r = TEMPLATE_SIZE // 2

    # 1. Extract Anchor Template
    x1, x2 = max(0, int(x_anc)-r), min(w, int(x_anc)+r+1)
    y1, y2 = max(0, int(y_anc)-r), min(h, int(y_anc)+r+1)
    template = img_anchor[y1:y2, x1:x2]
    
    if template.size == 0 or template.shape[0] < r or template.shape[1] < r:
        return global_guess # Template invalid (too close to edge)

    # 2. Define RESTRICTED Search Region (Centered on Global Guess)
    # We only look 40px around where the WHOLE tissue moved.
    gx, gy = global_guess.ravel()
    sr = SEARCH_RESTRICTION
    
    sx1 = max(0, int(gx) - sr)
    sx2 = min(w, int(gx) + sr + template.shape[1])
    sy1 = max(0, int(gy) - sr)
    sy2 = min(h, int(gy) + sr + template.shape[0])
    
    search_region = img_target[sy1:sy2, sx1:sx2]
    
    if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
        return global_guess # Search region invalid

    # 3. Match
    res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # 4. THE VOID GUARD (Crucial Step)
    # Calculate candidate position
    top_left_x, top_left_y = max_loc
    cand_x = sx1 + top_left_x + (template.shape[1] // 2)
    cand_y = sy1 + top_left_y + (template.shape[0] // 2)
    
    # Check if this new spot is pitch black
    # (Safe-guard bounds first)
    cx_int, cy_int = int(cand_x), int(cand_y)
    if 0 <= cx_int < w and 0 <= cy_int < h:
        pixel_val = img_target[cy_int, cx_int]
        
        if pixel_val < MIN_INTENSITY:
            # REJECT! The match landed in the background.
            # Fallback strategy: Stick to the Global Guess (which is safer)
            return global_guess
            
    # If intensity is good, and correlation is decent
    if max_val > 0.4:
        return np.array([cand_x, cand_y], dtype=np.float32)
    
    return global_guess # Match failed, trust Global Shift

def process_core(core_id, anchor_idx, manual_points, core_folder):
    slice_files = sorted(glob(os.path.join(core_folder, "*.ome.tif")))
    trajectories = {}
    anchor_name = os.path.basename(slice_files[anchor_idx])
    
    anchor_points_fixed = np.array(manual_points, dtype=np.float32)
    
    # Track "Last Valid" just for relative movement, but we mainly use Anchor
    last_points = anchor_points_fixed.copy()
    
    trajectories[anchor_name] = anchor_points_fixed.tolist()
    
    img_anchor_orig = load_image_ck(slice_files[anchor_idx])
    if img_anchor_orig is None: return {}

    # --- FORWARD ---
    img_prev = img_anchor_orig.copy() # For global shift calculation
    
    for i in range(anchor_idx + 1, len(slice_files)):
        img_target_orig = load_image_ck(slice_files[i])
        if img_target_orig is None: break
        
        # Pad
        img_anc_pad, img_tgt_pad = pad_to_match(img_anchor_orig, img_target_orig)
        img_prev_pad, _ = pad_to_match(img_prev, img_target_orig) # For global shift
        
        # 1. Calculate Global Shift (Prev -> Curr)
        dx, dy = get_global_shift(img_prev_pad, img_tgt_pad)
        
        new_points = []
        for idx, pt_anchor in enumerate(anchor_points_fixed):
            # Predict location based on Last Known + Global Shift
            last_pt = last_points[idx]
            global_guess = last_pt + np.array([dx, dy], dtype=np.float32)
            
            # Refine using Guarded Template Match (Anchor -> Target)
            pt_out = track_guarded(img_anc_pad, img_tgt_pad, pt_anchor, global_guess)
            new_points.append(pt_out)
            
        last_points = np.array(new_points, dtype=np.float32)
        trajectories[os.path.basename(slice_files[i])] = last_points.tolist()
        img_prev = img_target_orig # Update prev for next global shift calc

    # --- BACKWARD ---
    # Reset
    last_points = anchor_points_fixed.copy()
    img_prev = img_anchor_orig.copy()
    
    for i in range(anchor_idx - 1, -1, -1):
        img_target_orig = load_image_ck(slice_files[i])
        if img_target_orig is None: break
        
        img_anc_pad, img_tgt_pad = pad_to_match(img_anchor_orig, img_target_orig)
        img_prev_pad, _ = pad_to_match(img_prev, img_target_orig)
        
        dx, dy = get_global_shift(img_prev_pad, img_tgt_pad)
        
        new_points = []
        for idx, pt_anchor in enumerate(anchor_points_fixed):
            last_pt = last_points[idx]
            global_guess = last_pt + np.array([dx, dy], dtype=np.float32)
            
            pt_out = track_guarded(img_anc_pad, img_tgt_pad, pt_anchor, global_guess)
            new_points.append(pt_out)
            
        last_points = np.array(new_points, dtype=np.float32)
        trajectories[os.path.basename(slice_files[i])] = last_points.tolist()
        img_prev = img_target_orig

    return trajectories

def run_propagation():
    print(f"Loading Manual Landmarks from {LANDMARK_FILE}...")
    if not os.path.exists(LANDMARK_FILE): return
    df = pd.read_csv(LANDMARK_FILE)
    final_database = {}
    
    for core_id, group in tqdm(df.groupby("Core_ID")):
        core_folder = os.path.join(INPUT_ROOT, core_id)
        if not os.path.exists(core_folder): continue
        try:
            res = process_core(core_id, int(group.iloc[0]['Slice_Index']), group[['X', 'Y']].values.tolist(), core_folder)
            final_database[core_id] = res
        except Exception as e: print(e)

    with open(OUTPUT_JSON, 'w') as f: json.dump(final_database, f, indent=4)
    print(f"Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    run_propagation()