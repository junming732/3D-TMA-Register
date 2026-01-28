import os
import cv2
import numpy as np
import tifffile
import config
import math

def rotate_image_and_points(image, points, angle_degrees):
    """ Rotates image and (cx, cy) points around the center. """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Note: We use negative angle here because OpenCV rotates CCW for positive.
    # If we detect a slope of +2 degrees (downward), we want to rotate -2 degrees to fix it.
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
    
    # Sort by X to make finding right-neighbors easier
    sorted_c = sorted(candidates, key=lambda c: c['cx'])
    
    for i, c1 in enumerate(sorted_c):
        # Look for the closest neighbor to the RIGHT
        best_dist = float('inf')
        best_angle = None
        
        for c2 in sorted_c[i+1:]:
            dx = c2['cx'] - c1['cx']
            dy = c2['cy'] - c1['cy']
            
            # Optimization: If dx is huge, stop checking (sorted list)
            if dx > 300: break 
            
            # We only care about immediate neighbors in the same row
            # If dy is too large relative to dx, it's likely a diagonal neighbor (next row)
            if abs(dy) > abs(dx): continue 
            
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Check if this is a reasonable neighbor distance (e.g. < 300 pixels)
            if dist < 300: 
                angle = np.degrees(np.arctan2(dy, dx))
                angles.append(angle)
                
    if not angles: return 0.0
    
    # Return the median angle. 
    # If rows slant down, angle is positive. We need to rotate counter-clockwise (positive) to fix? 
    # Wait, OpenCV rotation: Positive = Counter-Clockwise.
    # If slope is positive (downwards), we need to lift it UP (Counter-Clockwise).
    # So we return angle directly.
    return np.median(angles)

def generate_qc_maps_neighbor_fix():
    review_dir = os.path.join(config.WORKSPACE, "QC_Maps_Neighbor_Fix")
    os.makedirs(review_dir, exist_ok=True)
    
    print(f"--- Generating Maps (Nearest-Neighbor Rotation Fix) ---")
    print(f"Saving to: {review_dir}\n")

    for i, file_path in enumerate(config.TMA_FILES):
        tma_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        save_path = os.path.join(review_dir, f"{i+1:02d}_{tma_name}_Final.jpg")
        
        if not os.path.exists(file_path): continue

        # --- 1. PARAMETERS ---
        params = {
            "CLAHE_LIMIT": 3.0, "CLAHE_GRID": (10, 10),
            "OPEN_SIZE": 25, "THRESH_FACTOR": 0.87,    
            "MIN_AREA": 3500, "PADDING": 1.25,
            "MAX_CORES": 35, "SORT_METHOD": "AREA",    
            "KILL_BORDER": False
        }

        if "TMA_3" in tma_name: params["OPEN_SIZE"] = 35    
        elif "TMA_13" in tma_name:
            params["THRESH_FACTOR"] = 0.70; params["MIN_AREA"] = 1000
            params["SORT_METHOD"] = "CANNY_EDGES"; params["MAX_CORES"] = 30        
        elif "TMA_15" in tma_name:
            params["KILL_BORDER"] = True; params["MIN_AREA"] = 5000; params["MAX_CORES"] = 30        

        OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["OPEN_SIZE"], params["OPEN_SIZE"]))
        CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        try:
            with tifffile.TiffFile(file_path) as tif:
                series0 = tif.series[0]
                low_res = series0.levels[-1]
                if low_res.shape[-1] < 100: low_res = series0.levels[-2]
                
                raw_stack = low_res.asarray()
                combined = np.sum(raw_stack, axis=0, dtype=np.float32)
                p99 = np.percentile(combined, 99)
                if p99 < 1: p99 = combined.max()
                norm = np.clip((combined / p99) * 255.0, 0, 255).astype(np.uint8)

                clahe = cv2.createCLAHE(clipLimit=params["CLAHE_LIMIT"], tileGridSize=params["CLAHE_GRID"])
                enhanced = clahe.apply(norm)
                blur = cv2.GaussianBlur(enhanced, (15, 15), 0)
                otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, thresh = cv2.threshold(blur, otsu_val * params["THRESH_FACTOR"], 255, cv2.THRESH_BINARY)
                
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=2)
                final_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=2)
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if params["SORT_METHOD"] == "CANNY_EDGES": canny_map = cv2.Canny(norm, 50, 150)
                else: canny_map = None

                candidates = []
                img_h, img_w = norm.shape
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < params["MIN_AREA"]: continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = float(w)/h
                    if aspect < 0.3 or aspect > 3.0: continue
                    
                    M = cv2.moments(cnt)
                    if M["m00"] != 0: cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    else: cx, cy = x + w/2, y + h/2

                    if params["KILL_BORDER"]:
                        mw, mh = img_w * 0.05, img_h * 0.05
                        if (cx < mw) or (cx > img_w - mw) or (cy < mh) or (cy > img_h - mh): continue 

                    canny_score = 0
                    if canny_map is not None:
                        mask = np.zeros(norm.shape, np.uint8)
                        cv2.drawContours(mask, [cnt], -1, 255, -1)
                        internal_edges = cv2.bitwise_and(canny_map, canny_map, mask=mask)
                        canny_score = cv2.countNonZero(internal_edges) / (area + 1e-5)

                    candidates.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area, 'canny': canny_score})

                if not candidates: continue

                # --- 1. ROTATION CORRECTION (NEIGHBOR METHOD) ---
                detected_angle = get_row_based_rotation(candidates)
                print(f"    -> Detected Tilt: {detected_angle:.2f} degrees")
                
                # Safety Clamp: If the angle is > 10 degrees, something is wrong. Don't rotate wildly.
                if abs(detected_angle) > 10:
                    print("    -> WARNING: Tilt too high. Clamping to 0 (Safety).")
                    detected_angle = 0
                
                qc_img_base = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
                qc_img, rotated_candidates = rotate_image_and_points(qc_img_base, candidates, detected_angle)
                img_h, img_w = qc_img.shape[:2]

                # --- 2. SELECTION ---
                if params["SORT_METHOD"] == "CANNY_EDGES": rotated_candidates.sort(key=lambda k: k['canny'], reverse=True)
                else: rotated_candidates.sort(key=lambda k: k['area'], reverse=True)
                
                selection = rotated_candidates[:params["MAX_CORES"]]

                if selection:
                    median_w = np.median([c['w'] for c in selection])
                    median_h = np.median([c['h'] for c in selection])
                    standard_size = int(max(median_w, median_h) * params["PADDING"])
                else: standard_size = 100

                # --- 3. GAP-BASED ROW DETECTION ---
                y_coords = np.array([c['cy'] for c in selection])
                y_coords.sort()
                
                diffs = np.diff(y_coords)
                gap_threshold = median_h * 0.5 # A gap bigger than half a core means new row
                
                split_indices = np.where(diffs > gap_threshold)[0] + 1
                row_groups = np.split(y_coords, split_indices)
                
                rows = []
                for group in row_groups:
                    if len(group) == 0: continue
                    min_y, max_y = group.min(), group.max()
                    current_row_cores = [c for c in selection if min_y <= c['cy'] <= max_y]
                    current_row_cores.sort(key=lambda k: k['cx'])
                    rows.append(current_row_cores)

                # --- DRAW ---
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
    generate_qc_maps_neighbor_fix()