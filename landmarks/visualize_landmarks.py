"""
Landmark Propagation Quality Control (QC) Visualizer
====================================================

This module generates visual verification artifacts for the automated landmark 
propagation pipeline. It overlays the computed coordinate trajectories onto the 
original image stacks to allow human assessment of tracking stability.

Architectural Context
---------------------
Automated optical flow tracking can drift due to tissue tears or rapid deformation. 
Blindly trusting mathematical convergence metrics (TRE) is risky. This module 
provides the "Ground Truth Verification" layer by rendering the tracking results 
as animated GIFs, enabling the engineer to instantly spot-check if landmarks 
"stick" to their biological targets across the Z-stack.

Methodology
-----------
1. **Data Fusion:** Merges the raw 4x downsampled image data (CK Channel) with the 
   `propagated_landmarks.json` coordinate database.
2. **Visual Overlay:** Converts 16-bit grayscale medical images to 8-bit RGB and 
   renders high-contrast markers (Green Dots) at the tracked coordinates.
3. **Temporal Rendering:** Compiles the annotated slices into a chronological GIF 
   animation (5 FPS), visualizing the tissue deformation and the corresponding 
   landmark movement in real-time.

Input / Output
--------------
* **Input:** 1. `propagated_landmarks.json` (from the Propagation Module).
    2. Directory of 4x Downsampled images.
* **Output:** A directory `QC_Visualizations/` containing one `.gif` file per core.
"""
import os
import cv2
import json
import imageio
import tifffile
import numpy as np
from glob import glob
from tqdm import tqdm
import config

# --- CONFIGURATION ---
INPUT_ROOT = INPUT_ROOT = os.path.join(config.DATASPACE, "downsampled_4x")
JSON_FILE = os.path.join(INPUT_ROOT, "propagated_landmarks.json")
OUTPUT_QC_DIR = os.path.join(INPUT_ROOT, "QC_Visualizations")
CK_INDEX = 6  # Index of Cytokeratin channel

# Visualization Settings
DOT_COLOR = (0, 255, 0)  # Green
DOT_RADIUS = 4           # Visible size on 4x images
FPS = 1                  # Frames Per Second for GIF

def pad_image_to_size(img, target_h, target_w):
    """
    Pads an image to exactly (target_h, target_w) using black borders.
    """
    h, w = img.shape[:2]
    
    # Calculate padding amounts
    pad_h = target_h - h
    pad_w = target_w - w
    
    # Safety: If image is somehow bigger than target (shouldn't happen with our logic), crop it
    if pad_h < 0 or pad_w < 0:
        return img[:target_h, :target_w]
    
    # Pad (Top, Bottom, Left, Right)
    # OpenCV uses order: top, bottom, left, right
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    return img_padded

def create_qc_gif(core_id, core_data, output_path):
    """
    Generates a GIF showing the landmarks overlaid on the tissue stack.
    Handles variable image sizes by padding to the largest slice dimensions.
    """
    core_folder = os.path.join(INPUT_ROOT, core_id)
    images = []
    
    # 1. Sort files to ensure Z-order is correct
    sorted_filenames = sorted(core_data.keys())
    
    # --- PASS 1: FIND MAX DIMENSIONS ---
    max_h, max_w = 0, 0
    valid_files = [] # Keep track of files that actually exist
    
    for fname in sorted_filenames:
        fpath = os.path.join(core_folder, fname)
        if not os.path.exists(fpath): continue
        
        try:
            # Quick check of dimensions without full decode if possible, 
            # but tifffile.imread is safer for OME-TIFF
            img = tifffile.imread(fpath)
            if img.shape[0] < 20: 
                # (C, H, W) -> extract H, W
                h, w = img.shape[1], img.shape[2]
            else:
                # (H, W, C) -> extract H, W
                h, w = img.shape[0], img.shape[1]
                
            if h > max_h: max_h = h
            if w > max_w: max_w = w
            
            valid_files.append(fname)
            
        except Exception as e:
            print(f"Error checking dims for {fname}: {e}")

    if max_h == 0 or max_w == 0:
        print(f"Skipping {core_id}: No valid images found.")
        return False

    # --- PASS 2: GENERATE FRAMES ---
    for fname in valid_files:
        fpath = os.path.join(core_folder, fname)
        
        try:
            # Load CK Channel
            img = tifffile.imread(fpath)
            if img.shape[0] < 20: ck = img[CK_INDEX, :, :]
            else: ck = img[:, :, CK_INDEX]
            
            # Normalize to 8-bit for RGB conversion
            ck_norm = cv2.normalize(ck, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Convert to RGB
            ck_rgb = cv2.cvtColor(ck_norm, cv2.COLOR_GRAY2RGB)
            
            # --- ARCHITECTURAL FIX: PAD TO MAX SIZE ---
            ck_final = pad_image_to_size(ck_rgb, max_h, max_w)
            
            # Draw Landmarks
            # Note: Coordinates are relative to Top-Left, so padding Right/Bottom doesn't break them.
            points = core_data[fname]
            for x, y in points:
                if np.isnan(x) or np.isnan(y): continue
                
                center = (int(x), int(y))
                # Bounds check before drawing
                if 0 <= center[0] < max_w and 0 <= center[1] < max_h:
                    cv2.circle(ck_final, center, DOT_RADIUS, DOT_COLOR, -1)
            
            # Add Text Label
            cv2.putText(ck_final, fname, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            images.append(ck_final)
            
        except Exception as e:
            print(f"Error rendering frame {fname}: {e}")

    # Save GIF
    if images:
        try:
            imageio.mimsave(output_path, images, fps=FPS, loop=0)
            return True
        except Exception as e:
            print(f"Error saving GIF for {core_id}: {e}")
            return False
            
    return False

def run_visualization():
    print("="*60)
    print("  LANDMARK QC VISUALIZER (GIF Generator)")
    print("="*60)
    
    # 1. Load Data
    with open(JSON_FILE, 'r') as f:
        database = json.load(f)
        
    os.makedirs(OUTPUT_QC_DIR, exist_ok=True)
    
    print(f"Found {len(database)} cores to visualize.")
    
    # 2. Process
    for core_id, core_data in tqdm(database.items()):
        out_name = os.path.join(OUTPUT_QC_DIR, f"{core_id}_QC.gif")
        create_qc_gif(core_id, core_data, out_name)
        
    print(f"\nQC Complete. Check folder: {OUTPUT_QC_DIR}")

if __name__ == "__main__":
    run_visualization()