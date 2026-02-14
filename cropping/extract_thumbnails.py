"""
Macro/Label Image Extraction & Enhancement
==========================================

This module is a utility script designed to extract the "thumbnail" image 
from whole-slide OME-TIFF files. 

Raw macro images stored in slide scanners are often dark, low-contrast, or have 
incorrect color profiles. This script automatically corrects channel ordering and 
applies adaptive histogram equalization to ensure the slide label (barcode, ID) 
is clearly legible.

Key Features:
-------------
1. Series 1 Extraction: Targets the specific IFD index typically reserved for the 
   slide label/macro view in OME-TIFF standards.
2. Color Space Management: 
   - Handles Channel-First (CHW) to Channel-Last (HWC) conversion.
   - Converts RGB to BGR for OpenCV compatibility.
3. Adaptive Enhancement (CLAHE): 
   - Converts images to LAB color space.
   - Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the 
     L-channel (Lightness) only.
   - This improves local contrast and brightness without oversaturating or 
     distorting the color information.

Output:
    - Saves enhanced JPEGs to: `[WORKSPACE]/Thumbnails_Macro_Bright/`
    - Filename format: `[Index]_[TMA_Name]_Macro.jpg`
"""
import os
import cv2
import numpy as np
import tifffile
import config  # Imports your config.py

def extract_bright_thumbnails():
    gallery_dir = os.path.join(config.WORKSPACE, "Thumbnails_Macro_Bright")
    os.makedirs(gallery_dir, exist_ok=True)
    
    print(f"--- Extracting & Brightening 'Series 1' Thumbnails ---")
    print(f"Saving to: {gallery_dir}\n")

    for i, file_path in enumerate(config.TMA_FILES):
        tma_folder_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        save_name = f"{i+1:02d}_{tma_folder_name}_Macro.jpg"
        output_path = os.path.join(gallery_dir, save_name)

        if not os.path.exists(file_path):
            continue
        
        try:
            with tifffile.TiffFile(file_path) as tif:
                if len(tif.series) > 1:
                    # 1. Get raw data
                    image_data = tif.series[1].asarray()
                    
                    # 2. Fix Color Channel Order (RGB -> BGR for OpenCV)
                    if image_data.ndim == 3:
                        if image_data.shape[0] == 3: # (3, H, W) -> (H, W, 3)
                            image_data = np.moveaxis(image_data, 0, -1)
                        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

                    # 3. AUTO-BRIGHTNESS (The Fix)
                    # We convert to Lab color space to brighten ONLY the "Lightness" channel
                    # This prevents colors from getting weirdly saturated
                    lab = cv2.cvtColor(image_data, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    # This is smarter than simple stretching; it enhances details locally.
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    
                    # Merge back
                    limg = cv2.merge((cl, a, b))
                    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                    # 4. Save
                    cv2.imwrite(output_path, final)
                    print(f"[{i+1}] Saved Enhanced: {save_name}")
                    
                else:
                    print(f"Error: No thumbnail series for {tma_folder_name}")

        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    extract_bright_thumbnails()