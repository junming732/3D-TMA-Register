"""
TMA Core Detection Visualization
================================

This module generates visual inspection maps for Tissue Microarray (TMA) slides. 
It functions as a diagnostic tool to visualize the precise output of the core 
detection algorithms before running the heavy-duty extraction pipeline.

The script processes whole-slide images to detect tissue cores, applies the same 
segmentation logic as the extraction pipeline, and overlays the results (bounding 
boxes, centroids, and IDs) onto a high-contrast JPEG map.

Key Features:
-------------
1. Visual Verification: Produces lightweight JPEG maps showing exactly what the algorithm sees.
2. Shared Logic: Uses the same "Safe Mask," "Triangle Thresholding," and "Linear Stretch" 
   techniques as the extraction module to ensure parity.
3. Rotation Feedback: Visualizes the detected grid rotation angle to verify alignment.


Output:
    - Saves annotated images to: `[WORKSPACE]/QC_Maps_Triangle_Linear/`
    - Filename format: `[Index]_[TMA_Name]_QC.jpg`

Usage:
    Run directly to generate maps for all slides defined in `config.TMA_FILES`.
"""

import os
import cv2
import numpy as np
import tifffile
import config  # Assumes your config.py is in the same folder
import math

# --- HELPER FUNCTIONS ---

def rotate_image_and_points(image, points, angle_degrees):
    """ Rotates image and (cx, cy) points around the center. """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    new_points = []
    points_array = np.array([[p['cx'], p['cy']] for p in points], dtype=np.float32)
    if len(points_array) > 0:
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points_array, ones])
        transformed = M.dot(points_ones.T).T
        for i, p in enumerate(points):
            new_p = p.copy()
            new_p['cx'] = transformed[i][0]
            new_p['cy'] = transformed[i][1]
            new_points.append(new_p)
    return rotated_img, new_points

def get_row_based_rotation(candidates):
    """
    Robust Angle Detection:
    Finds the angle between each core and its nearest right-side neighbor.
    The median of these angles is the true row tilt.
    """
    if len(candidates) < 2: return 0.0
    
    angles = []
    sorted_c = sorted(candidates, key=lambda c: c['cx'])
    
    for i, c1 in enumerate(sorted_c):
        for c2 in sorted_c[i+1:]:
            dx = c2['cx'] - c1['cx']
            dy = c2['cy'] - c1['cy']
            
            if dx > 300: break 
            if abs(dy) > abs(dx): continue 
            
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 300: 
                angle = np.degrees(np.arctan2(dy, dx))
                angles.append(angle)
                
    if not angles: return 0.0
    return np.median(angles)

def apply_linear_stretch(image, low_p=0.5, high_p=99.5):
    """
    Stretches histogram to full 0-255 range.
    low_p=0.5: heavily clips black noise (good for background)
    high_p=99.5: keeps bright signal
    """
    p_min, p_max = np.percentile(image[image > 0], (low_p, high_p))
    return np.clip((image - p_min) / (p_max - p_min + 1e-5) * 255, 0, 255).astype(np.uint8)

# --- MAIN PROCESSING FUNCTION ---

def generate_qc_maps_final():
    review_dir = os.path.join(config.WORKSPACE, "QC_Maps_Triangle_Linear")
    os.makedirs(review_dir, exist_ok=True)
    
    print(f"--- Generating Maps (Linear Stretch + Triangle Threshold) ---")
    print(f"Saving to: {review_dir}\n")

    # Parameters
    params = {
        "OPEN_SIZE": 15,         # Morphology cleanup size
        "MIN_AREA": 2000,        # Minimum core size (pixels)
        "PADDING": 1.35,         # QC Box size padding
        "MAX_CORES": 30,         # Max cores per slice
        "TILT_LIMIT": 12.0       # Max rotation angle allowed
    }

    OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["OPEN_SIZE"], params["OPEN_SIZE"]))
    CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

    for i, file_path in enumerate(config.TMA_FILES):
        tma_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        save_path = os.path.join(review_dir, f"{i+1:02d}_{tma_name}_QC.jpg")
        
        if not os.path.exists(file_path): continue

        try:
            with tifffile.TiffFile(file_path) as tif:
                # --- 1. Load Image ---
                series0 = tif.series[0]
                low_res = series0.levels[-1]
                if low_res.shape[-1] < 100: low_res = series0.levels[-2]
                
                raw_stack = low_res.asarray()
                combined = np.sum(raw_stack, axis=0, dtype=np.float32)
                
                # Basic Normalization (0-255)
                p99 = np.percentile(combined, 99)
                if p99 < 1: p99 = combined.max()
                norm = np.clip((combined / p99) * 255.0, 0, 255).astype(np.uint8)

                # --- 2. Generate "Safe Mask" (Background Removal) ---
                # We use aggressive subtraction ONLY to identify where tissue is
                kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
                bg_est = cv2.morphologyEx(norm, cv2.MORPH_OPEN, kernel_bg)
                foreground_rough = cv2.subtract(norm, bg_est)
                
                # Create a binary mask of "roughly where tissue is"
                _, rough_mask = cv2.threshold(foreground_rough, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Dilate heavily to ensure we don't clip edges of faint cores
                safe_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

                # --- 3. Contrast Enhancement (Linear Stretch) ---
                stretched_img = apply_linear_stretch(norm)

                # --- 4. Segmentation (Triangle Method) ---
                # CRITICAL: We calculate threshold on the UNMASKED image to let Triangle see the noise peak.
                blur = cv2.GaussianBlur(stretched_img, (15, 15), 0)
                thresh_val, binary_raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
                
                # NOW apply the Safe Mask to the binary result
                binary_masked = cv2.bitwise_and(binary_raw, binary_raw, mask=safe_mask)
                
                # Morphology Cleanup
                closed = cv2.morphologyEx(binary_masked, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=2)
                final_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=2)
                
                # --- 5. Find Candidates ---
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidates = []
                img_h, img_w = norm.shape
                
                # === SPECIAL HANDLING FOR SLICE 13 ===
                if (i + 1) == 3:
                    print(f"[{i+1}] {tma_name}: Applying SPECIAL ARTIFACT SPLITTING (Kernel 50)...")
                    SPLIT_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        
                        # CASE A: Monster Blob (Likely Worm + Cores)
                        if area > 20000:
                            # Isolate this blob
                            blob_mask = np.zeros_like(norm)
                            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
                            
                            # Aggressively Erode to break connections
                            eroded_blob = cv2.erode(blob_mask, SPLIT_KERNEL, iterations=2)
                            
                            # Find pieces inside
                            sub_contours, _ = cv2.findContours(eroded_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for sub_cnt in sub_contours:
                                sub_area = cv2.contourArea(sub_cnt)
                                if sub_area < 500: continue # Ignore dust
                                
                                # Dilate back to restore size
                                piece_mask = np.zeros_like(norm)
                                cv2.drawContours(piece_mask, [sub_cnt], -1, 255, -1)
                                restored_piece = cv2.dilate(piece_mask, SPLIT_KERNEL, iterations=2)
                                
                                # Get final contour of restored piece
                                restored_cnts, _ = cv2.findContours(restored_piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if restored_cnts:
                                    final_cnt = restored_cnts[0]
                                    final_area = cv2.contourArea(final_cnt)
                                    
                                    x, y, w, h = cv2.boundingRect(final_cnt)
                                    M = cv2.moments(final_cnt)
                                    if M["m00"] != 0:
                                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                                        candidates.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': final_area})
                        
                        # CASE B: Normal Core
                        elif area > params["MIN_AREA"]:
                            x, y, w, h = cv2.boundingRect(cnt)
                            aspect = float(w)/h
                            if aspect < 0.3 or aspect > 3.0: continue
                            
                            M = cv2.moments(cnt)
                            if M["m00"] != 0: 
                                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                            else: 
                                cx, cy = x + w/2, y + h/2
                            candidates.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area})

                # === STANDARD HANDLING FOR ALL OTHER SLICES ===
                else:
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < params["MIN_AREA"]: continue
                        
                        x, y, w, h = cv2.boundingRect(cnt)
                        aspect = float(w)/h
                        if aspect < 0.3 or aspect > 3.0: continue
                        
                        M = cv2.moments(cnt)
                        if M["m00"] != 0: 
                            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        else: 
                            cx, cy = x + w/2, y + h/2

                        # Optional: Border Killing
                        # mw, mh = img_w * 0.05, img_h * 0.05
                        # if (cx < mw) or (cx > img_w - mw) or (cy < mh) or (cy > img_h - mh): continue 

                        candidates.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area})

                if not candidates:
                    print(f"[{i+1}] {tma_name}: No candidates found.")
                    continue

                # --- 6. Rotation Correction ---
                detected_angle = get_row_based_rotation(candidates)
                
                # TILT LIMITATION: 12 DEGREES
                if abs(detected_angle) > params["TILT_LIMIT"]:
                    print(f"    -> Tilt {detected_angle:.2f}° too high. Clamping to 0.")
                    detected_angle = 0
                else:
                    print(f"    -> Tilt {detected_angle:.2f}° accepted.")
                
                qc_img_base = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
                qc_img, rotated_candidates = rotate_image_and_points(qc_img_base, candidates, detected_angle)
                img_h, img_w = qc_img.shape[:2]

                # --- 7. Sort & Grid Logic ---
                # Sort by Area (Largest first)
                rotated_candidates.sort(key=lambda k: k['area'], reverse=True)
                selection = rotated_candidates[:params["MAX_CORES"]]

                if selection:
                    # Get Y-coords
                    y_coords = np.array([c['cy'] for c in selection])
                    y_coords.sort()
                    
                    # Split rows based on gaps
                    diffs = np.diff(y_coords)
                    median_h = np.median([c['h'] for c in selection])
                    gap_threshold = median_h * 0.5
                    
                    split_indices = np.where(diffs > gap_threshold)[0] + 1
                    row_groups = np.split(y_coords, split_indices)
                    
                    rows = []
                    for group in row_groups:
                        if len(group) == 0: continue
                        min_y, max_y = group.min(), group.max()
                        current_row_cores = [c for c in selection if min_y <= c['cy'] <= max_y]
                        current_row_cores.sort(key=lambda k: k['cx']) # Sort Left-to-Right
                        rows.append(current_row_cores)

                    # Draw Final Map
                    standard_size = int(max(median_h, np.median([c['w'] for c in selection])) * params["PADDING"])
                    count = 0
                    
                    for row in rows:
                        for c in row:
                            count += 1
                            label = str(count)
                            
                            x = int(c['cx'] - standard_size/2)
                            y = int(c['cy'] - standard_size/2)
                            x = max(0, min(x, img_w - standard_size))
                            y = max(0, min(y, img_h - standard_size))

                            cv2.rectangle(qc_img, (x, y), (x+standard_size, y+standard_size), (0, 255, 0), 2)
                            cv2.circle(qc_img, (int(c['cx']), int(c['cy'])), 5, (0, 0, 255), -1)
                            cv2.putText(qc_img, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imwrite(save_path, qc_img)
                print(f"[{i+1}] Saved: {tma_name} (Found: {count})")

        except Exception as e:
            print(f"ERROR {tma_name}: {e}")

if __name__ == "__main__":
    generate_qc_maps_final()