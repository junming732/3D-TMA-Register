import os
import cv2
import numpy as np
import tifffile
import config
import math
import zarr

# --- 1. ROTATION LOGIC (UNCHANGED) ---

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
    """Robust angle detection."""
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

# --- 2. MAIN PIPELINE (GROUP BY CORE) ---

def crop_and_group_by_core():
    # 1. Output to DATASPACE
    output_base = os.path.join(config.DATASPACE, "TMA_Cores_Grouped")
    os.makedirs(output_base, exist_ok=True)
    
    print(f"--- Starting Cropping (All Slices -> Grouped by Core) ---")
    print(f"Output: {output_base}\n")

    PIXEL_SIZE_UM = 0.4961
    CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

    # 2. Loop through ALL slices
    for i, file_path in enumerate(config.TMA_FILES):
        tma_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        
        print(f"[{i+1}/{len(config.TMA_FILES)}] Processing Slide: {tma_name}...")
        
        if not os.path.exists(file_path): 
            print(f"   -> File not found: {file_path}")
            continue

        params = {
            "CLAHE_LIMIT": 3.0, "CLAHE_GRID": (10, 10),
            "OPEN_SIZE": 25, 
            "THRESH_FACTOR": 0.70,   
            "MIN_AREA": 1000,
            "PADDING": 1.25,
            "MAX_CORES": 30,
            "SORT_METHOD": "CANNY_EDGES",
            "KILL_BORDER": False
        }

        try:
            with tifffile.TiffFile(file_path) as tif:
                series0 = tif.series[0]
                level_0 = series0.levels[0]
                low_res = series0.levels[-1]
                if low_res.shape[-1] < 100: low_res = series0.levels[-2]
                h_high, w_high = level_0.shape[-2], level_0.shape[-1]
                h_low, w_low = low_res.shape[-2], low_res.shape[-1]
                scale_x = w_high / w_low
                scale_y = h_high / h_low

                raw_stack = low_res.asarray()
                if raw_stack.shape[0] < 20: 
                    combined = np.sum(raw_stack, axis=0, dtype=np.float32)
                else: 
                    combined = np.sum(raw_stack, axis=2, dtype=np.float32)

                p99 = np.percentile(combined, 99)
                if p99 < 1: p99 = combined.max()
                norm = np.clip((combined / p99) * 255.0, 0, 255).astype(np.uint8)
                
                clahe = cv2.createCLAHE(clipLimit=params["CLAHE_LIMIT"], tileGridSize=params["CLAHE_GRID"])
                enhanced = clahe.apply(norm)
                blur = cv2.GaussianBlur(enhanced, (15, 15), 0)
                otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, thresh = cv2.threshold(blur, otsu_val * params["THRESH_FACTOR"], 255, cv2.THRESH_BINARY)
                
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)
                final_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["OPEN_SIZE"], params["OPEN_SIZE"])), iterations=2)
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if params["SORT_METHOD"] == "CANNY_EDGES": 
                    canny_map = cv2.Canny(norm, 50, 150)
                else: 
                    canny_map = None

                candidates = []
                img_h, img_w = norm.shape
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
                    
                    canny_score = 0
                    if canny_map is not None:
                        mask = np.zeros(norm.shape, np.uint8)
                        cv2.drawContours(mask, [cnt], -1, 255, -1)
                        internal_edges = cv2.bitwise_and(canny_map, canny_map, mask=mask)
                        canny_score = cv2.countNonZero(internal_edges) / (area + 1e-5)
                    candidates.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area, 'canny': canny_score})

                if not candidates: 
                    print("   -> No candidates found.")
                    continue

                # --- ROTATION CORRECTION ---
                detected_angle = get_row_based_rotation(candidates)
                
                if abs(detected_angle) > 10:
                    detected_angle = 0
                
                dummy_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                _, rotated_candidates = rotate_image_and_points(dummy_img, candidates, detected_angle)

                # --- SELECTION ---
                if params["SORT_METHOD"] == "CANNY_EDGES": 
                    rotated_candidates.sort(key=lambda k: k['canny'], reverse=True)
                else: 
                    rotated_candidates.sort(key=lambda k: k['area'], reverse=True)
                
                selection = rotated_candidates[:params["MAX_CORES"]]

                median_w = np.median([c['w'] for c in selection])
                median_h = np.median([c['h'] for c in selection])
                low_res_box_size = int(max(median_w, median_h) * params["PADDING"])

                # --- ROW DETECTION ---
                y_coords = np.array([c['cy'] for c in selection])
                y_coords.sort()
                
                diffs = np.diff(y_coords)
                gap_threshold = median_h * 0.5
                
                split_indices = np.where(diffs > gap_threshold)[0] + 1
                row_groups_y = np.split(y_coords, split_indices)
                
                rows = []
                for group_y in row_groups_y:
                    if len(group_y) == 0: continue
                    min_y, max_y = group_y.min(), group_y.max()
                    current_row_cores = [c for c in selection if min_y <= c['cy'] <= max_y]
                    current_row_cores.sort(key=lambda k: k['cx'])
                    rows.append(current_row_cores)
                
                ordered_cores = []
                for row in rows:
                    ordered_cores.extend(row)

                # --- EXTRACTION ---
                store = level_0.aszarr()
                z = zarr.open(store, mode='r')
                if isinstance(z, zarr.Group):
                    if '0' in z: z = z['0']
                    else: z = z[list(z.keys())[0]]

                is_channel_first = True
                if z.ndim == 3 and z.shape[0] > z.shape[2]: 
                    is_channel_first = False
                
                center = (img_w // 2, img_h // 2)
                M_inv = cv2.getRotationMatrix2D(center, -detected_angle, 1.0)
                
                for idx, core in enumerate(ordered_cores):
                    core_id = idx + 1
                    
                    # 3. FOLDER STRUCTURE: .../TMA_Cores_Grouped/Core_01/
                    core_dir = os.path.join(output_base, f"Core_{core_id:02d}")
                    os.makedirs(core_dir, exist_ok=True)
                    
                    rotated_point = np.array([[core['cx'], core['cy']]], dtype=np.float32)
                    ones = np.ones(shape=(1, 1))
                    point_ones = np.hstack([rotated_point, ones])
                    original_point = M_inv.dot(point_ones.T).T
                    orig_cx, orig_cy = original_point[0][0], original_point[0][1]
                    
                    high_cx, high_cy = int(orig_cx * scale_x), int(orig_cy * scale_y)
                    high_box = int(low_res_box_size * scale_x)
                    half_box = high_box // 2
                    
                    start_x, start_y = max(0, high_cx - half_box), max(0, high_cy - half_box)
                    end_x, end_y = min(w_high, high_cx + half_box), min(h_high, high_cy + half_box)
                    
                    if is_channel_first: 
                        crop = z[:, start_y:end_y, start_x:end_x]
                    else: 
                        crop = z[start_y:end_y, start_x:end_x, :]
                        crop = np.transpose(crop, (2, 0, 1))

                    crop = np.ascontiguousarray(crop, dtype=np.uint16)
                    c_dim, h_dim, w_dim = crop.shape

                    # FILENAME: TMA_Name_CoreXX.tif (inside the Core_XX folder)
                    out_name = f"{tma_name}_Core{core_id:02d}.tif"
                    out_full = os.path.join(core_dir, out_name)
                    
                    # --- SAVING LOGIC (ImageJ Style) ---
                    tifffile.imwrite(
                        out_full,
                        crop,
                        imagej=True,  
                        photometric='minisblack',
                        resolution=(1.0 / PIXEL_SIZE_UM, 1.0 / PIXEL_SIZE_UM),
                        metadata={
                            'axes': 'CYX',         
                            'unit': 'um',          
                            'spacing': 0.0,
                            'images': c_dim,
                            'slices': 1,
                            'frames': 1,
                            'hyperstack': True,
                            'mode': 'composite',
                            'Labels': CHANNEL_NAMES 
                        },
                        compression=None
                    )
                    
                    # --- THUMBNAIL LOGIC ---
                    thumb_raw = crop[0, :, :] 
                    p99_thumb = np.percentile(thumb_raw, 99)
                    if p99_thumb == 0: p99_thumb = thumb_raw.max() if thumb_raw.max() > 0 else 1
                    thumb_norm = np.clip((thumb_raw / p99_thumb) * 255.0, 0, 255).astype(np.uint8)
                    new_h = 512
                    new_w = int((w_dim / h_dim) * new_h)
                    thumb_resized = cv2.resize(thumb_norm, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Save thumbnail in the SAME Core_XX folder
                    cv2.imwrite(os.path.join(core_dir, f"{tma_name}_Core{core_id:02d}_thumb.jpg"), thumb_resized)
                    
                print(f"    -> Successfully grouped {len(ordered_cores)} cores into 'Core_XX' folders.\n")

        except Exception as e:
            print(f"ERROR {tma_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    crop_and_group_by_core()