"""
Core Alignment Visualization (GIF Generator)
============================================

This module provides a visual verification tool for 3D TMA registration. It compiles 
sequential image slices of extracted cores into animated GIFs, allowing users to 
rapidly assess the spatial alignment and stability of the tissue stack.

The script iterates through processed core directories, naturally sorts the image 
sequences (e.g., Slice_1 -> Slice_2 -> Slice_10), and generates a looped animation 
with overlayed text labels.

Key Features:
-------------
1. Natural Sorting: Ensures frames are ordered numerically (1, 2, 10) rather 
   than lexicographically (1, 10, 2).
2. Dynamic Standardization: Automatically resizes all subsequent frames to match 
   the dimensions of the first slice, preventing GIF corruption from varying input sizes.
3. Text Overlay: Burns the filename/slice ID into each frame for easy identification 
   of specific problematic layers.
4. Format Agnostic: Supports mixing .jpg, .png, and .tif inputs.

Output:
    - Saves GIFs to: `[BASE_DIR]/_GIF_Inspection/`
    - Filename format: `[Core_Name]_alignment.gif`

Usage:
    Configure `BASE_DIR` in the script to point to your image dataset and run.
"""

import os
import imageio.v3 as iio
from natsort import natsorted
import glob
import cv2  # OpenCV for resizing and drawing text
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = "/work/junming/3D-TMA-Register/Core_thumb_NEW"
OUTPUT_DIR = os.path.join(BASE_DIR, "_GIF_Inspection")
FRAME_DURATION = 0.5  # 0.5 seconds per frame

# Text Settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0       # Adjust if text is too big/small
TEXT_POS = (15, 40)    # coordinates (x, y) for top-left
THICKNESS = 2
COLOR_FG = (255, 255, 255) # White text
COLOR_BG = (0, 0, 0)       # Black outline
# ---------------------

def create_gifs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sort Core folders (Core_1, Core_2, Core_10...)
    core_folders = natsorted([
        f for f in os.listdir(BASE_DIR) 
        if os.path.isdir(os.path.join(BASE_DIR, f)) and "Core" in f
    ])

    print(f"Found {len(core_folders)} core folders.")

    for core_name in core_folders:
        core_path = os.path.join(BASE_DIR, core_name)
        
        # 1. Collect all images
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            images.extend(glob.glob(os.path.join(core_path, ext)))
        
        # 2. CRITICAL: Natural Sort
        images = natsorted(images)

        if not images:
            continue

        # Debug print to confirm order
        first_file = os.path.basename(images[0])
        last_file = os.path.basename(images[-1])
        print(f"Processing {core_name} ({len(images)} slices)... [{first_file} -> {last_file}]")

        frames = []
        target_shape = None  # Will lock to the size of Slice_1

        for img_path in images:
            try:
                # --- READ & STANDARDIZE IMAGE ---
                img = iio.imread(img_path)

                # Convert Grayscale to RGB if needed (H, W) -> (H, W, 3)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                # Drop Alpha if needed (H, W, 4) -> (H, W, 3)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

                # --- RESIZE TO MATCH FIRST FRAME ---
                # Set target size based on the FIRST slice found
                if target_shape is None:
                    target_shape = (img.shape[1], img.shape[0]) # (Width, Height)
                
                # Resize current slice if it doesn't match Slice_1 dimensions
                if (img.shape[1], img.shape[0]) != target_shape:
                    img = cv2.resize(img, target_shape, interpolation=cv2.INTER_LINEAR)

                # --- [NEW] ADD TITLE TEXT ---
                # Get filename without extension (e.g., "Slice_05")
                title_text = os.path.splitext(os.path.basename(img_path))[0]

                # 1. Draw thicker black outline for readability
                cv2.putText(img, title_text, TEXT_POS, FONT, FONT_SCALE, 
                            COLOR_BG, THICKNESS + 3, cv2.LINE_AA)
                
                # 2. Draw thinner white text on top
                cv2.putText(img, title_text, TEXT_POS, FONT, FONT_SCALE, 
                            COLOR_FG, THICKNESS, cv2.LINE_AA)

                frames.append(img)

            except Exception as e:
                print(f"  Warning: Skipping bad frame {img_path}: {e}")

        # --- SAVE GIF ---
        if len(frames) > 1:
            save_path = os.path.join(OUTPUT_DIR, f"{core_name}_alignment.gif")
            
            # Using imageio v3 API with pillow backend for GIF
            # Duration is in milliseconds. loop=0 means infinite.
            try:
                # Try v3 standard way
                iio.imwrite(save_path, frames, duration=FRAME_DURATION * 1000, loop=0)
            except TypeError:
                 # Fallback for older versions/backends that might expect seconds
                iio.imwrite(save_path, frames, duration=FRAME_DURATION, loop=0)
                
            print(f"  -> Saved GIF: {core_name}_alignment.gif")

    print("\nAll done! Check '_GIF_Inspection' folder.")

if __name__ == "__main__":
    create_gifs()