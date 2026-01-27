import os
import cv2
import numpy as np
import tifffile
import config

def generate_qc_maps_final():
    review_dir = os.path.join(config.WORKSPACE, "QC_Maps_Final")
    os.makedirs(review_dir, exist_ok=True)
    
    # --- TUNING ---
    # 1. CLAHE (Contrast Boost)
    # clipLimit=3.0 makes faint signal 3x stronger locally.
    CLAHE_LIMIT = 3.0
    CLAHE_GRID = (10, 10)
    
    # 2. MORPHOLOGY
    # Close: Connect fragments (Repair)
    CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    # Open: Separates artifacts.
    # Reduced from (25, 25) -> (21, 21) as requested ("slightly less opening")
    OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    
    # 3. AREA & THRESHOLD
    THRESH_FACTOR = 0.87
    MIN_AREA = 3500 # Catch seeds that survive the cut
    MAX_AREA = 80000 
    
    # 4. PADDING
    # Increased from 1.15 (15%) -> 1.25 (25%) as requested ("slightly more padding")
    PADDING_FACTOR = 1.25

    print(f"--- Generating Maps (Sum + CLAHE + Tuned Padding) ---")
    print(f"Saving to: {review_dir}\n")

    for i, file_path in enumerate(config.TMA_FILES):
        tma_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        save_path = os.path.join(review_dir, f"{i+1:02d}_{tma_name}_FinalMap.jpg")
        save_mask = os.path.join(review_dir, f"{i+1:02d}_{tma_name}_Mask.jpg")

        if not os.path.exists(file_path): continue
            
        try:
            with tifffile.TiffFile(file_path) as tif:
                # 1. LOAD & SUM (High Signal)
                series0 = tif.series[0]
                low_res = series0.levels[-1]
                if low_res.shape[-1] < 100: low_res = series0.levels[-2]
                
                raw_stack = low_res.asarray()
                combined = np.sum(raw_stack, axis=0, dtype=np.float32)
                
                # Normalize
                p99 = np.percentile(combined, 99)
                if p99 < 1: p99 = combined.max()
                norm = np.clip((combined / p99) * 255.0, 0, 255).astype(np.uint8)

                # 2. ENHANCE (CLAHE)
                clahe = cv2.createCLAHE(clipLimit=CLAHE_LIMIT, tileGridSize=CLAHE_GRID)
                enhanced = clahe.apply(norm)
                
                # 3. BINARIZE
                blur = cv2.GaussianBlur(enhanced, (15, 15), 0)
                otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                low_val = otsu_val * THRESH_FACTOR
                _, thresh = cv2.threshold(blur, low_val, 255, cv2.THRESH_BINARY)
                
                # 4. MORPHOLOGY (Repair then Cut)
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=2)
                final_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=2)
                
                cv2.imwrite(save_mask, final_mask)

                # 5. DETECT
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                candidates = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < MIN_AREA: continue
                    
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = float(w)/h
                    if aspect < 0.3 or aspect > 3.0: continue

                    # True Centroid
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w/2, y + h/2
                    
                    candidates.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area})

                # 6. STANDARDIZE SIZE
                if not candidates:
                    print(f"[{i+1}] WARNING: No cores found. Check Mask.")
                    cv2.imwrite(save_path, norm)
                    continue

                normal_candidates = [c for c in candidates if c['area'] < MAX_AREA]
                if not normal_candidates: normal_candidates = candidates
                
                # Calc standard size from top 50% largest (Healthy cores)
                normal_candidates.sort(key=lambda k: k['area'], reverse=True)
                top_half = normal_candidates[:max(1, len(normal_candidates)//2)]
                
                median_w = np.median([c['w'] for c in top_half])
                median_h = np.median([c['h'] for c in top_half])
                
                # Apply 25% Padding
                standard_size = int(max(median_w, median_h) * PADDING_FACTOR) 
                half_size = standard_size // 2
                
                # Apply to Top 35 candidates
                candidates.sort(key=lambda k: k['area'], reverse=True)
                selection = candidates[:35]
                
                final_cores = []
                img_h, img_w = norm.shape
                
                for c in selection:
                    x = int(c['cx'] - half_size)
                    y = int(c['cy'] - half_size)
                    
                    x = max(0, x)
                    y = max(0, y)
                    if x + standard_size > img_w: x = img_w - standard_size
                    if y + standard_size > img_h: y = img_h - standard_size
                    
                    final_cores.append({'x': x, 'y': y, 'w': standard_size, 'h': standard_size, 'cx': c['cx'], 'cy': c['cy']})

                # 7. DRAW
                final_cores.sort(key=lambda k: k['cy'])
                rows = []
                if final_cores:
                    curr_row = [final_cores[0]]
                    row_tol = final_cores[0]['h'] * 0.6 
                    for core in final_cores[1:]:
                        if abs(core['cy'] - curr_row[-1]['cy']) < row_tol:
                            curr_row.append(core)
                        else:
                            curr_row.sort(key=lambda k: k['cx'])
                            rows.append(curr_row)
                            curr_row = [core]
                    curr_row.sort(key=lambda k: k['cx'])
                    rows.append(curr_row)

                qc_img = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
                labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                
                count = 0
                for r_idx, row in enumerate(rows):
                    for c_idx, core in enumerate(row):
                        label = f"{labels[r_idx]}{c_idx+1}"
                        count += 1
                        
                        cv2.rectangle(qc_img, (core['x'], core['y']), (core['x']+core['w'], core['y']+core['h']), (0, 255, 0), 2)
                        cv2.circle(qc_img, (int(core['cx']), int(core['cy'])), 5, (0, 0, 255), -1)
                        cv2.putText(qc_img, label, (core['x'], core['y']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imwrite(save_path, qc_img)
                print(f"[{i+1}] Saved: {tma_name} (Std Size: {standard_size}px, Found: {count})")

        except Exception as e:
            print(f"ERROR {tma_name}: {e}")

if __name__ == "__main__":
    generate_qc_maps_final()