import os
import cv2
import numpy as np
import tifffile
import config
import math
import zarr

# --- 1. HELPER FUNCTIONS ---

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
    """Angle detection."""
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

# --- 2. MAIN PIPELINE ---

def crop_and_group_by_core():
    # Output to DATASPACE
    output_base = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_NEW")
    os.makedirs(output_base, exist_ok=True)
    
    print("=" * 70)
    print("  TMA CORE EXTRACTION")
    print("=" * 70)
    print(f"Output directory: {output_base}")
    print(f"Total slides to process: {len(config.TMA_FILES)}\n")

    PIXEL_SIZE_UM = 0.4961
    CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

    # Statistics
    total_cores_extracted = 0
    successful_slides = 0
    failed_slides = []

    # Process all slides
    for i, file_path in enumerate(config.TMA_FILES):
        tma_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        
        print(f"\n{'─' * 70}")
        print(f"[{i+1}/{len(config.TMA_FILES)}] Processing: {tma_name}")
        print(f"{'─' * 70}")
        
        if not os.path.exists(file_path): 
            print(f"File not found: {file_path}")
            failed_slides.append((tma_name, "File not found"))
            continue

        # --- PARAMETERS (MATCHING QC SCRIPT) ---
        params = {
            "OPEN_SIZE": 15,         # Morphology cleanup size
            "MIN_AREA": 2000,        # Minimum core size (pixels)
            "PADDING": 1.35,         # QC Box size padding
            "MAX_CORES": 30,         # Max cores per slice
            "TILT_LIMIT": 12.0       # Max rotation angle allowed
        }

        # KERNELS
        OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["OPEN_SIZE"], params["OPEN_SIZE"]))
        CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

        try:
            with tifffile.TiffFile(file_path) as tif:
                # --- LOAD IMAGE ---
                series0 = tif.series[0]
                level_0 = series0.levels[0]
                low_res = series0.levels[-1]
                if low_res.shape[-1] < 100: low_res = series0.levels[-2]
                
                # Calculate Scaling for High Res extraction later
                h_high, w_high = level_0.shape[-2], level_0.shape[-1]
                h_low, w_low = low_res.shape[-2], low_res.shape[-1]
                scale_x = w_high / w_low
                scale_y = h_high / h_low

                # Prepare Low Res for Detection
                raw_stack = low_res.asarray()
                combined = np.sum(raw_stack, axis=0, dtype=np.float32)

                # Basic Normalization
                p99 = np.percentile(combined, 99)
                if p99 < 1: p99 = combined.max()
                norm = np.clip((combined / p99) * 255.0, 0, 255).astype(np.uint8)
                
                # --- PREPROCESSING ---
                
                # 1. Generate "Safe Mask" (Background Removal)
                kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
                bg_est = cv2.morphologyEx(norm, cv2.MORPH_OPEN, kernel_bg)
                foreground_rough = cv2.subtract(norm, bg_est)
                _, rough_mask = cv2.threshold(foreground_rough, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                safe_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

                # 2. Contrast Enhancement (Linear Stretch)
                stretched_img = apply_linear_stretch(norm)

                # 3. Segmentation (Triangle Method on UNMASKED image)
                blur = cv2.GaussianBlur(stretched_img, (15, 15), 0)
                thresh_val, binary_raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
                
                # 4. Apply Safe Mask
                binary_masked = cv2.bitwise_and(binary_raw, binary_raw, mask=safe_mask)
                
                # 5. Morphology
                closed = cv2.morphologyEx(binary_masked, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=2)
                final_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=2)
                
                # --- FIND CANDIDATES ---
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidates = []
                img_h, img_w = norm.shape
                
                # === SPECIAL HANDLING FOR SLICE 3 ===
                if (i + 1) == 3:
                    print(f"   Applying SPECIAL SLICE 3 LOGIC (Monster Splitter)...")
                    SPLIT_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        
                        # CASE A: Monster Blob
                        if area > 20000:
                            blob_mask = np.zeros_like(norm)
                            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
                            eroded_blob = cv2.erode(blob_mask, SPLIT_KERNEL, iterations=2)
                            sub_contours, _ = cv2.findContours(eroded_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for sub_cnt in sub_contours:
                                sub_area = cv2.contourArea(sub_cnt)
                                if sub_area < 500: continue
                                
                                piece_mask = np.zeros_like(norm)
                                cv2.drawContours(piece_mask, [sub_cnt], -1, 255, -1)
                                restored_piece = cv2.dilate(piece_mask, SPLIT_KERNEL, iterations=2)
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

                # === STANDARD HANDLING ===
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
                        
                        candidates.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area})

                if not candidates: 
                    print(" No candidates found")
                    failed_slides.append((tma_name, "No candidates found"))
                    continue

                # --- ROTATION CORRECTION ---
                detected_angle = get_row_based_rotation(candidates)
                
                # TILT LIMITATION: 12 DEGREES
                if abs(detected_angle) > params["TILT_LIMIT"]:
                    print(f"  Rotation: {detected_angle:.2f}° (CLIPPED to 0° > Limit {params['TILT_LIMIT']}°)")
                    detected_angle = 0
                else:
                    print(f"  Rotation: {detected_angle:.2f}°")
                
                # Apply rotation to candidates
                dummy_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                _, rotated_candidates = rotate_image_and_points(dummy_img, candidates, detected_angle)

                # --- SELECTION & SORTING ---
                # Sort by Area (Largest first)
                rotated_candidates.sort(key=lambda k: k['area'], reverse=True)
                selection = rotated_candidates[:params["MAX_CORES"]]

                if selection:
                    median_w = np.median([c['w'] for c in selection])
                    median_h = np.median([c['h'] for c in selection])
                    standard_size = int(max(median_w, median_h) * params["PADDING"])
                else: 
                    standard_size = 100
                
                low_res_box_size = standard_size

                # --- ROW DETECTION (GRID LOGIC) ---
                y_coords = np.array([c['cy'] for c in selection])
                y_coords.sort()
                diffs = np.diff(y_coords)
                gap_threshold = median_h * 0.5
                split_indices = np.where(diffs > gap_threshold)[0] + 1
                row_groups = np.split(y_coords, split_indices)
                
                rows = []
                for group in row_groups:
                    if len(group) == 0: continue
                    min_y, max_y = group.min(), group.max()
                    current_row_cores = [c for c in selection if min_y <= c['cy'] <= max_y]
                    current_row_cores.sort(key=lambda k: k['cx'])
                    rows.append(current_row_cores)
                
                print(f"  Detected: {len(rows)} rows, {len(selection)} cores total")

                ordered_cores = []
                for row in rows:
                    ordered_cores.extend(row)

                # --- EXTRACTION (HIGH RES) ---
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
                
                print(f"  Extracting cores... ", end="", flush=True)
                
                for idx, core in enumerate(ordered_cores):
                    core_id = idx + 1
                    
                    core_dir = os.path.join(output_base, f"Core_{core_id:02d}")
                    os.makedirs(core_dir, exist_ok=True)
                    
                    # Map low-res rotated point back to original High-Res coordinates
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

                    out_name = f"{tma_name}_Core{core_id:02d}.ome.tif"
                    out_full = os.path.join(core_dir, out_name)
                    
                    # OME-TIFF METADATA
                    metadata = {
                        'axes': 'CYX',
                        'Channel': {'Name': CHANNEL_NAMES},
                        'PhysicalSizeX': PIXEL_SIZE_UM,
                        'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeY': PIXEL_SIZE_UM,
                        'PhysicalSizeYUnit': 'µm',
                    }
                    
                    tifffile.imwrite(
                        out_full,
                        crop,
                        photometric='minisblack',
                        metadata=metadata,
                        compression=None
                    )
                    
                    # THUMBNAIL
                    thumb_raw = crop[0, :, :] 
                    p99_thumb = np.percentile(thumb_raw, 99)
                    if p99_thumb == 0: p99_thumb = thumb_raw.max() if thumb_raw.max() > 0 else 1
                    thumb_norm = np.clip((thumb_raw / p99_thumb) * 255.0, 0, 255).astype(np.uint8)
                    new_h = 512
                    new_w = int((w_dim / h_dim) * new_h)
                    thumb_resized = cv2.resize(thumb_norm, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    cv2.imwrite(os.path.join(core_dir, f"{tma_name}_Core{core_id:02d}_thumb.jpg"), thumb_resized)
                    
                    # Progress indicator
                    if (core_id) % 5 == 0 or core_id == len(ordered_cores):
                        print(f"{core_id}", end="", flush=True)
                        if core_id != len(ordered_cores):
                            print("...", end="", flush=True)
                
                print(f" Complete")
                print(f" Saved {len(ordered_cores)} cores to Core_XX/ folders")
                
                total_cores_extracted += len(ordered_cores)
                successful_slides += 1

        except Exception as e:
            print(f" ERROR: {e}")
            failed_slides.append((tma_name, str(e)))
            import traceback
            traceback.print_exc()

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 70)
    print("  EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Successful slides:    {successful_slides}/{len(config.TMA_FILES)}")
    print(f"Total cores extracted: {total_cores_extracted}")
    print(f"Output directory:      {output_base}")
    
    if failed_slides:
        print(f"\nFailed slides ({len(failed_slides)}):")
        for tma_name, reason in failed_slides:
            print(f"  • {tma_name}: {reason}")
    


if __name__ == "__main__":
    crop_and_group_by_core()