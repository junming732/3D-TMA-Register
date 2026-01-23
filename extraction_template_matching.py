import tifffile
import numpy as np
import os
from skimage.feature import match_template
from config import TMA_FOLDERS, WORKSPACE

# --- INPUTS ---
PIXEL_SIZE_L0 = 0.4961
X_MICRONS = 1668.8
Y_MICRONS = 1557.0
SIZE_MICRONS = 1200 

# --- SETTINGS ---
SEARCH_LEVEL = 4     # Low res for speed
SAVE_LEVEL = 2       # Med res for quality
SEARCH_RADIUS = 150  # Only look 150 pixels (low-res) away from center
MAX_JUMP = 200       # If it moves more than this, it's wrong.

# Calculate Base Coordinates (Level 4)
search_scale = 2**SEARCH_LEVEL 
save_scale = 2**SAVE_LEVEL     

# Expected Center (Low Res)
cx_base = int((X_MICRONS / PIXEL_SIZE_L0) / search_scale)
cy_base = int((Y_MICRONS / PIXEL_SIZE_L0) / search_scale)
box_r_low = int((SIZE_MICRONS / PIXEL_SIZE_L0) / search_scale / 2)

def safe_crop(img, cy, cx, r):
    """Safely crops a fixed-size box (2r, 2r), padding with zeros if at edge."""
    h, w = img.shape[-2:] # Works for (C,H,W) or (H,W)
    pad_y1, pad_y2, pad_x1, pad_x2 = 0, 0, 0, 0
    
    y1, y2 = cy - r, cy + r
    x1, x2 = cx - r, cx + r
    
    # Calculate padding if out of bounds
    if y1 < 0: pad_y1 = -y1; y1 = 0
    if x1 < 0: pad_x1 = -x1; x1 = 0
    if y2 > h: pad_y2 = y2 - h; y2 = h
    if x2 > w: pad_x2 = x2 - w; x2 = w
    
    # Crop based on dimensionality
    if img.ndim == 3:
        crop = img[:, y1:y2, x1:x2]
        # Pad (Channels, Y, X)
        crop = np.pad(crop, ((0,0), (pad_y1, pad_y2), (pad_x1, pad_x2)))
    else:
        crop = img[y1:y2, x1:x2]
        crop = np.pad(crop, ((pad_y1, pad_y2), (pad_x1, pad_x2)))
        
    return crop

def leashed_harvest():
    print(f"--- LEASHED SEARCH: Center=({cx_base}, {cy_base}) Radius={SEARCH_RADIUS} ---")
    
    # 1. ACQUIRE TEMPLATE (From Slice 1)
    template = None
    path1 = os.path.join(TMA_FOLDERS[0], "Scan1", f"{os.path.basename(TMA_FOLDERS[0])}_Scan1.unmixed.qptiff")
    
    with tifffile.TiffFile(path1) as tif:
        page = tif.series[0].levels[SEARCH_LEVEL]
        full_low = page.asarray()
        if full_low.ndim == 3: full_low = full_low[0] # DAPI only
        
        # Crop exactly at the user coordinate
        template = safe_crop(full_low, cy_base, cx_base, box_r_low)
        
    print(f"Template Size: {template.shape}")

    stack = []
    
    # 2. PROCESS ALL SLICES
    for i, folder_path in enumerate(TMA_FOLDERS):
        try:
            fname = f"{os.path.basename(folder_path)}_Scan1.unmixed.qptiff"
            fpath = os.path.join(folder_path, "Scan1", fname)
            
            with tifffile.TiffFile(fpath) as tif:
                # A. Define Strictly Limited Search Area
                # We only load a small box around the EXPECTED center
                y_min = max(0, cy_base - SEARCH_RADIUS - box_r_low)
                y_max = cy_base + SEARCH_RADIUS + box_r_low
                x_min = max(0, cx_base - SEARCH_RADIUS - box_r_low)
                x_max = cx_base + SEARCH_RADIUS + box_r_low
                
                page_search = tif.series[0].levels[SEARCH_LEVEL]
                # Slicing here is safe because we computed limits above
                # But we need to handle the dimension
                if page_search.ndim == 3: # Not supported by asarray sometimes?
                     # Safe load strategy: load full if small enough, or check shape
                     # For Level 4, loading full image is fast (20MB)
                     img_full = page_search.asarray()
                     if img_full.ndim == 3: img_full = img_full[0]
                else:
                     img_full = page_search.asarray()
                
                # Crop the search window
                search_window = img_full[y_min:y_max, x_min:x_max]

                # B. Match Template
                result = match_template(search_window, template)
                ij = np.unravel_index(np.argmax(result), result.shape)
                y_local, x_local = ij 
                
                # C. Convert Local Match -> Global Coordinate
                found_cx = x_min + x_local + box_r_low
                found_cy = y_min + y_local + box_r_low
                
                # D. SANITY CHECK (The Leash)
                # If it jumped too far from Slice 1, ignore it and use Slice 1's center
                dist = np.sqrt((found_cx - cx_base)**2 + (found_cy - cy_base)**2)
                
                if dist > MAX_JUMP:
                    print(f"Slice {i+1}: JUMP TOO BIG ({int(dist)}px). Resetting to center.")
                    final_cx_low = cx_base
                    final_cy_low = cy_base
                else:
                    print(f"Slice {i+1}: Locked at ({found_cx}, {found_cy})")
                    final_cx_low = found_cx
                    final_cy_low = found_cy

                # E. Extract High-Res (Level 2)
                scale = search_scale / save_scale # 4.0
                save_cx = int(final_cx_low * scale)
                save_cy = int(final_cy_low * scale)
                save_r = int((SIZE_MICRONS / PIXEL_SIZE_L0) / save_scale / 2)
                
                # Load Level 2
                page_save = tif.series[0].levels[SAVE_LEVEL]
                img_save = page_save.asarray()
                
                # USE SAFE CROP (Prevents shape errors)
                crop = safe_crop(img_save, save_cy, save_cx, save_r)
                stack.append(crop)

        except Exception as e:
            print(f"Slice {i+1} Failed: {e}")
            # Append a blank frame to keep the stack consistent? 
            # Better to skip or add noise. For now, we skip.

    # 3. SAVE
    if len(stack) > 0:
        # Check shapes
        print(f"Stacking {len(stack)} images. Shape of first: {stack[0].shape}")
        output_path = os.path.join(WORKSPACE, "robust_stack.npy")
        np.save(output_path, np.array(stack))
        print(f"SUCCESS: Saved to {output_path}")

if __name__ == "__main__":
    leashed_harvest()