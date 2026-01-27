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