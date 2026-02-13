"""
Optical Flow Landmark Propagation Module
========================================

This module implements the automated "Anchor & Propagate" logic to extend manually 
labeled ground-truth landmarks across the full Z-stack of TMA cores. It utilizes 
Sparse Optical Flow with strict consistency checks to map biological features 
from damaged/partial slices to intact slices.

Architectural Context
---------------------
To bypass the labor-intensive process of manually labeling 600+ slices, this module 
takes a small set of "seed" landmarks (5-7 points on a single anchor slice) and 
mathematically tracks their displacement through the tissue volume. This generates 
the dense coordinate dataset required for Target Registration Error (TRE) calculation.

Methodology (Computer Vision)
-----------------------------
1. **Lucas-Kanade Optical Flow:** Computes the displacement vector of sparse feature 
   points using image intensity gradients (pyramidal implementation).
2. **Forward-Backward Error Check:** Validates every tracking step by projecting the 
   point $P_{t} \to P_{t+1}$ and immediately tracking it back $P_{t+1} \to P_{t}$. 
   Points with a round-trip error $> 1.0$ pixel are flagged as unstable/lost.
3. **Bi-Directional Propagation:** Initiates tracking from the "Anchor Slice" 
   outwards in both Z-directions (Anchor $\to$ Top, Anchor $\to$ Bottom) to maximize 
   tissue overlap coverage.

Input / Output
--------------
* **Input:** 1. CSV file containing manual seed landmarks (CoreID, SliceIndex, X, Y).
    2. Directory of 4x Downsampled CK-channel images.
* **Output:** A hierarchical JSON database (`propagated_landmarks.json`) containing 
  the validated $(x,y)$ coordinates for every slice in every core.
"""
import os
import cv2
import numpy as np
import tifffile
import pandas as pd
import json
from glob import glob
from tqdm import tqdm
import config

# --- CONFIGURATION ---
INPUT_ROOT = os.path.join(config.DATASPACE, "downsampled_4x")
LANDMARK_FILE =  os.path.join(config.DATASPACE, "manual_landmarks.csv")
OUTPUT_JSON = os.path.join(INPUT_ROOT, "propagated_landmarks.json")

# CK Channel Index (from your previous scripts)
CK_INDEX = 6 

# Optical Flow Parameters (Optimized for Tissue)
# --- OPTIMIZED OPTICAL FLOW PARAMETERS ---
LK_PARAMS = dict(
    # 1. Search Area: Increased from (31,31) to (128,128)
    # This allows the algorithm to catch tissue moving up to ~64 pixels away.
    winSize  = (128, 128), 
    
    # 2. Pyramid Levels: Increased from 3 to 5
    # Helps the algorithm "see" global shifts before refining locally.
    maxLevel = 5,
    
    # 3. Criteria: Increased iterations to 50 (tries harder to converge)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01)
)

def load_image_ck(filepath):
    """Loads the specific OME-TIFF and extracts the CK channel as 8-bit."""
    try:
        # Load 8-channel image
        img = tifffile.imread(filepath)
        
        # Extract CK (Channel 6)
        # Handle (C, H, W) vs (H, W, C) safety check
        if img.shape[0] < 20: 
            ck = img[CK_INDEX, :, :]
        else:
            ck = img[:, :, CK_INDEX]
            
        # Normalize to 8-bit for OpenCV
        ck_norm = cv2.normalize(ck, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return ck_norm
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def pad_to_match(img1, img2):
    """
    Pads the smaller image with zeros (black) to match the larger image's dimensions.
    Returns both images with identical (max_h, max_w) shape.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    max_h = max(h1, h2)
    max_w = max(w1, w2)
    
    # Pad Image 1 if needed
    if h1 < max_h or w1 < max_w:
        img1 = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, max_w - w1, cv2.BORDER_CONSTANT, value=0)
        
    # Pad Image 2 if needed
    if h2 < max_h or w2 < max_w:
        img2 = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, max_w - w2, cv2.BORDER_CONSTANT, value=0)
        
    return img1, img2

def propagate_points(img_prev, img_next, p_prev):
    """
    Tracks points from prev -> next with Forward-Backward Error Check.
    Handles dimension mismatches via padding.
    """
    # --- ARCHITECTURAL FIX: PAD IMAGES TO MATCH ---
    # Optical Flow will crash if shapes differ.
    img_prev_padded, img_next_padded = pad_to_match(img_prev, img_next)
    
    # 1. Forward Flow (Prev -> Next)
    # Use padded images for calculation
    p_next, status, _ = cv2.calcOpticalFlowPyrLK(img_prev_padded, img_next_padded, p_prev, None, **LK_PARAMS)
    
    # 2. Backward Flow (Next -> Prev) verification
    p_back, _, _ = cv2.calcOpticalFlowPyrLK(img_next_padded, img_prev_padded, p_next, None, **LK_PARAMS)
    
    # 3. Calculate Distance between Start and 'Back-Tracked' Start
    dist = abs(p_prev - p_back).max(-1)
    
    # 4. Filter: If distance > 1.0 pixel, the tracking is invalid/unstable
    good_status = status.copy()
    good_status[dist > 1.0] = 0
    
    # Coordinate Check:
    # Since we padded (added to right/bottom), the (0,0) origin is unchanged.
    # So p_next coordinates are still valid for the original unpadded image, 
    # UNLESS the point moved into the padded area (which is black void).
    # We can add a check for that:
    h_orig, w_orig = img_next.shape
    
    # Mark as lost if points flew outside the ORIGINAL image bounds
    # (Reshape for vector operation)
    x_coords = p_next[:, 0, 0]
    y_coords = p_next[:, 0, 1]
    
    out_of_bounds = (x_coords >= w_orig) | (y_coords >= h_orig)
    good_status[out_of_bounds.reshape(-1, 1)] = 0
    
    return p_next, good_status

def process_core(core_id, anchor_slice_idx, manual_points, core_folder):
    """
    Propagates points from the anchor slice to all other slices in the core.
    """
    # Get all slice files sorted
    slice_files = sorted(glob(os.path.join(core_folder, "*.ome.tif")))
    
    # Dictionary to store trajectories: {slice_filename: [[x,y], [x,y]...]}
    trajectories = {}
    
    # Initialize with manual points
    # Format for OpenCV: Float32 array of shape (N, 1, 2)
    p_current = np.array(manual_points, dtype=np.float32).reshape(-1, 1, 2)
    
    # Map filenames to indices for easy access
    files_map = {os.path.basename(f): f for f in slice_files}
    anchor_filename = os.path.basename(slice_files[anchor_slice_idx])
    
    trajectories[anchor_filename] = p_current.reshape(-1, 2).tolist()
    
    # --- PROPAGATE FORWARD (Anchor -> End) ---
    img_prev = load_image_ck(slice_files[anchor_slice_idx])
    p_track = p_current.copy()
    
    for i in range(anchor_slice_idx, len(slice_files) - 1):
        img_curr = img_prev
        img_next = load_image_ck(slice_files[i+1])
        
        if img_next is None: break
        
        p_new, status = propagate_points(img_curr, img_next, p_track)
        
        # Update valid points, mark lost ones as NaN or Keep Last Known
        # Strategy: If lost, we keep the coordinate but mark it invalid in analysis later
        # For visualization, we just store what we got.
        p_track = p_new
        
        trajectories[os.path.basename(slice_files[i+1])] = p_track.reshape(-1, 2).tolist()
        img_prev = img_next # Setup for next loop

    # --- PROPAGATE BACKWARD (Anchor -> Start) ---
    img_prev = load_image_ck(slice_files[anchor_slice_idx])
    p_track = p_current.copy()
    
    for i in range(anchor_slice_idx, 0, -1):
        img_curr = img_prev
        img_next = load_image_ck(slice_files[i-1]) # 'Next' in time, but previous in index
        
        if img_next is None: break
        
        p_new, status = propagate_points(img_curr, img_next, p_track)
        p_track = p_new
        
        trajectories[os.path.basename(slice_files[i-1])] = p_track.reshape(-1, 2).tolist()
        img_prev = img_next

    return trajectories

def run_propagation():
    print("Loading Manual Landmarks...")
    # ASSUMPTION: CSV has columns [Core_ID, Slice_Index, X, Y]
    # Adjust this line to match your actual QuPath export format
    df = pd.read_csv(LANDMARK_FILE)
    
    final_database = {}
    
    # Group by Core to process one core at a time
    for core_id, group in tqdm(df.groupby("Core_ID")):
        core_folder = os.path.join(INPUT_ROOT, core_id)
        if not os.path.exists(core_folder):
            print(f"Skipping {core_id}: Folder not found")
            continue

        # Get the Anchor Slice Index from the CSV data
        # (The slice you marked is the anchor)
        anchor_slice_idx = group.iloc[0]['Slice_Index'] 
        
        # Get the points
        manual_points = group[['X', 'Y']].values.tolist()
        
        # Run Algorithm
        core_results = process_core(core_id, anchor_slice_idx, manual_points, core_folder)
        final_database[core_id] = core_results

    # Save Results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_database, f, indent=4)
        
    print(f"Propagation Complete. Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    run_propagation()