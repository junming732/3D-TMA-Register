"""
TMA Core Pipeline — Extract, Conform, Mask
==========================================

Combines what used to be three separate scripts into one, run as three
sequential phases so tissue masks are computed AFTER conforming instead of
before. This is simpler and inherently exact: the mask is generated directly
from the final canonical-shape image, so there's no crop/pad "carry-through"
math needed to keep an earlier mask pixel-aligned with a later reshape.

PHASE 1 — extract_cores_from_slides()
    Detects and extracts individual TMA cores from each whole-slide image,
    corrects grid tilt, writes each core/slide as a full-res multi-channel
    OME-TIFF into STAGING_BASE/Core_XX/, plus a JPEG thumbnail for QC.
    (Same detection/extraction logic as the original cropping script — no
    mask is generated here anymore.)

PHASE 2 — conform_all_cores()
    For each Core_XX folder, finds the canonical (H, W) shape (most common
    across that core's slices) and centre-crops/zero-pads any mismatched
    slice to match, writing the result to FINAL_BASE/Core_XX/. Slices that
    already match are copied as-is.

PHASE 3 — generate_masks_for_conformed()
    For each conformed OME-TIFF in FINAL_BASE, sums all channels, percentile-
    normalises, and runs the same background-subtract/Otsu/triangle-threshold
    morphology recipe used elsewhere in this pipeline, saving the result as
    a sibling '<stem>_tissue_mask.png'. Because this runs on the already-
    conformed image, the mask is exact by construction — same array, same
    shape, no alignment math to get right.

Downstream registration scripts (e.g. akaze_l0_channel_comparison.py) load
these masks directly via a '<stem>_tissue_mask.png' sibling-file lookup and
only fall back to recomputing if the file is missing or shape-mismatched.

Usage:
    python crop_conform_mask_tma.py                       # run all 3 phases
    python crop_conform_mask_tma.py --skip_extract         # already extracted
    python crop_conform_mask_tma.py --skip_extract --skip_conform   # masks only
    python crop_conform_mask_tma.py --dry_run              # phase 2/3, no writes
"""

import os
import sys
import math
import glob
import shutil
import logging
import argparse
from collections import Counter

import numpy as np
import cv2
import tifffile
import zarr

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

STAGING_BASE = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate")             # Phase 1 output / Phase 2 input
FINAL_BASE   = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")   # Phase 2 & 3 output

PIXEL_SIZE_UM = 0.4961
CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

parser = argparse.ArgumentParser(description="Combined TMA extract/conform/mask pipeline.")
parser.add_argument('--skip_extract', action='store_true', help='Skip phase 1 (extraction).')
parser.add_argument('--skip_conform', action='store_true', help='Skip phase 2 (conforming).')
parser.add_argument('--skip_mask',    action='store_true', help='Skip phase 3 (mask generation).')
parser.add_argument('--dry_run',      action='store_true', help='Phases 2/3: log actions, write nothing.')
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_mask_sibling_path(tif_path: str) -> str:
    """'<stem>.ome.tif' -> '<stem>_tissue_mask.png'. Explicit suffix strip
    rather than os.path.splitext, which mishandles the double '.ome.tif'
    extension (would incorrectly split to '<stem>.ome' + '.tif')."""
    if tif_path.endswith(".ome.tif"):
        stem = tif_path[:-len(".ome.tif")]
    else:
        stem = os.path.splitext(tif_path)[0]
    return stem + "_tissue_mask.png"


def load_as_chw(path: str) -> np.ndarray:
    """Load an .ome.tif and return it as (C, H, W)."""
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.moveaxis(arr, -1, 0)   # (H, W, C) -> (C, H, W)
    return arr


def apply_linear_stretch(image, low_p=0.5, high_p=99.5):
    """Stretch histogram to 0-255. low_p clips black noise, high_p keeps signal."""
    p_min, p_max = np.percentile(image[image > 0], (low_p, high_p))
    return np.clip((image - p_min) / (p_max - p_min + 1e-5) * 255, 0, 255).astype(np.uint8)


def compute_tissue_mask_fullres(vol_chw: np.ndarray, target_max_dim: float = 512.0) -> np.ndarray:
    """
    Tissue mask for one core's full multi-channel volume: sum channels ->
    percentile normalise -> background-subtract/Otsu safe-mask -> linear
    stretch -> blur -> triangle threshold -> AND safe mask -> close/open.
    Same recipe as the whole-slide core-detection mask in phase 1, applied
    per-core. Returns uint8 (H, W): 255 = tissue, 0 = background.
    """
    combined = np.sum(vol_chw.astype(np.float32), axis=0)
    p99 = np.percentile(combined, 99)
    if p99 < 1:
        p99 = combined.max() if combined.max() > 0 else 1.0
    norm = np.clip((combined / p99) * 255.0, 0, 255).astype(np.uint8)

    orig_h, orig_w = norm.shape
    scale = target_max_dim / max(orig_h, orig_w)
    img = (cv2.resize(norm, (int(orig_w * scale), int(orig_h * scale)),
                      interpolation=cv2.INTER_AREA) if scale < 1.0 else norm)

    kernel_bg   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
    bg_est      = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_bg)
    foreground  = cv2.subtract(img, bg_est)
    _, rough    = cv2.threshold(foreground, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    safe_mask   = cv2.morphologyEx(rough, cv2.MORPH_DILATE, kernel_safe)

    nonzero = img[img > 0]
    if len(nonzero) == 0:
        result = np.zeros_like(img)
    else:
        p_min, p_max  = np.percentile(nonzero, (0.5, 99.5))
        stretched     = np.clip((img.astype(np.float32) - p_min) / (p_max - p_min + 1e-5) * 255,
                                0, 255).astype(np.uint8)
        blur          = cv2.GaussianBlur(stretched, (15, 15), 0)
        _, binary_raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        binary_masked = cv2.bitwise_and(binary_raw, binary_raw, mask=safe_mask)
        close_kern    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        open_kern     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed        = cv2.morphologyEx(binary_masked, cv2.MORPH_CLOSE, close_kern, iterations=2)
        result        = cv2.morphologyEx(closed,        cv2.MORPH_OPEN,  open_kern,  iterations=2)

    if scale < 1.0:
        result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return result


def rotate_image_and_points(image, points, angle_degrees):
    """Rotates image and (cx, cy) points around the center."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
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
    """Angle detection based on nearest-neighbour row alignment."""
    if len(candidates) < 2:
        return 0.0
    angles = []
    sorted_c = sorted(candidates, key=lambda c: c['cx'])
    for i, c1 in enumerate(sorted_c):
        for c2 in sorted_c[i + 1:]:
            dx = c2['cx'] - c1['cx']
            dy = c2['cy'] - c1['cy']
            if dx > 300:
                break
            if abs(dy) > abs(dx):
                continue
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 300:
                angles.append(np.degrees(np.arctan2(dy, dx)))
    return np.median(angles) if angles else 0.0


def conform_slice(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Centre-crop or symmetrically zero-pad (C, H, W) to (C, target_h, target_w)."""
    c, h, w = arr.shape
    out    = np.zeros((c, target_h, target_w), dtype=arr.dtype)
    src_y0 = max(0, (h - target_h) // 2)
    dst_y0 = max(0, (target_h - h) // 2)
    copy_h = min(h - src_y0, target_h - dst_y0)
    src_x0 = max(0, (w - target_w) // 2)
    dst_x0 = max(0, (target_w - w) // 2)
    copy_w = min(w - src_x0, target_w - dst_x0)
    out[:, dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
        arr[:, src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — EXTRACT & ROTATE CORES FROM WHOLE-SLIDE IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def extract_cores_from_slides():
    os.makedirs(STAGING_BASE, exist_ok=True)
    logger.info("=" * 70)
    logger.info("PHASE 1 — TMA CORE EXTRACTION")
    logger.info("=" * 70)
    logger.info(f"Output directory: {STAGING_BASE}")
    logger.info(f"Total slides to process: {len(config.TMA_FILES)}")

    total_cores_extracted = 0
    successful_slides = 0
    failed_slides = []

    for i, file_path in enumerate(config.TMA_FILES):
        tma_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        logger.info(f"[{i+1}/{len(config.TMA_FILES)}] Processing: {tma_name}")

        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            failed_slides.append((tma_name, "File not found"))
            continue

        params = {
            "OPEN_SIZE": 15, "MIN_AREA": 2000, "PADDING": 1.35,
            "MAX_CORES": 30, "TILT_LIMIT": 12.0,
        }
        OPEN_KERNEL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["OPEN_SIZE"],) * 2)
        CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

        try:
            with tifffile.TiffFile(file_path) as tif:
                series0 = tif.series[0]
                level_0 = series0.levels[0]
                low_res = series0.levels[-1]
                if low_res.shape[-1] < 100:
                    low_res = series0.levels[-2]

                h_high, w_high = level_0.shape[-2], level_0.shape[-1]
                h_low, w_low   = low_res.shape[-2], low_res.shape[-1]
                scale_x = w_high / w_low
                scale_y = h_high / h_low

                raw_stack = low_res.asarray()
                combined  = np.sum(raw_stack, axis=0, dtype=np.float32)
                p99 = np.percentile(combined, 99)
                if p99 < 1:
                    p99 = combined.max()
                norm = np.clip((combined / p99) * 255.0, 0, 255).astype(np.uint8)

                kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
                bg_est = cv2.morphologyEx(norm, cv2.MORPH_OPEN, kernel_bg)
                foreground_rough = cv2.subtract(norm, bg_est)
                _, rough_mask = cv2.threshold(foreground_rough, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                safe_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_DILATE,
                                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

                stretched_img = apply_linear_stretch(norm)
                blur = cv2.GaussianBlur(stretched_img, (15, 15), 0)
                _, binary_raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
                binary_masked = cv2.bitwise_and(binary_raw, binary_raw, mask=safe_mask)
                closed = cv2.morphologyEx(binary_masked, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=2)
                final_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=2)

                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidates = []
                img_h, img_w = norm.shape

                if (i + 1) == 3:
                    logger.info("   Applying SPECIAL SLICE 3 LOGIC (Monster Splitter)...")
                    SPLIT_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 20000:
                            blob_mask = np.zeros_like(norm)
                            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
                            eroded_blob = cv2.erode(blob_mask, SPLIT_KERNEL, iterations=2)
                            sub_contours, _ = cv2.findContours(eroded_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for sub_cnt in sub_contours:
                                sub_area = cv2.contourArea(sub_cnt)
                                if sub_area < 500:
                                    continue
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
                        elif area > params["MIN_AREA"]:
                            x, y, w, h = cv2.boundingRect(cnt)
                            aspect = float(w) / h
                            if aspect < 0.3 or aspect > 3.0:
                                continue
                            M = cv2.moments(cnt)
                            cx, cy = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] != 0 else (x+w/2, y+h/2)
                            candidates.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area})
                else:
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < params["MIN_AREA"]:
                            continue
                        x, y, w, h = cv2.boundingRect(cnt)
                        aspect = float(w) / h
                        if aspect < 0.3 or aspect > 3.0:
                            continue
                        M = cv2.moments(cnt)
                        cx, cy = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] != 0 else (x+w/2, y+h/2)
                        candidates.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'area': area})

                if not candidates:
                    logger.warning("No candidates found")
                    failed_slides.append((tma_name, "No candidates found"))
                    continue

                detected_angle = get_row_based_rotation(candidates)
                if abs(detected_angle) > params["TILT_LIMIT"]:
                    logger.info(f"  Rotation: {detected_angle:.2f}° (CLIPPED to 0°, limit {params['TILT_LIMIT']}°)")
                    detected_angle = 0
                else:
                    logger.info(f"  Rotation: {detected_angle:.2f}°")

                dummy_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                _, rotated_candidates = rotate_image_and_points(dummy_img, candidates, detected_angle)

                rotated_candidates.sort(key=lambda k: k['area'], reverse=True)
                selection = rotated_candidates[:params["MAX_CORES"]]
                if selection:
                    median_w = np.median([c['w'] for c in selection])
                    median_h = np.median([c['h'] for c in selection])
                    standard_size = int(max(median_w, median_h) * params["PADDING"])
                else:
                    median_h = 100
                    standard_size = 100
                low_res_box_size = standard_size

                y_coords = np.array([c['cy'] for c in selection])
                y_coords.sort()
                diffs = np.diff(y_coords)
                gap_threshold = median_h * 0.5
                split_indices = np.where(diffs > gap_threshold)[0] + 1
                row_groups = np.split(y_coords, split_indices)

                rows = []
                for group in row_groups:
                    if len(group) == 0:
                        continue
                    min_y, max_y = group.min(), group.max()
                    current_row_cores = [c for c in selection if min_y <= c['cy'] <= max_y]
                    current_row_cores.sort(key=lambda k: k['cx'])
                    rows.append(current_row_cores)

                logger.info(f"  Detected: {len(rows)} rows, {len(selection)} cores total")

                ordered_cores = []
                for row in rows:
                    ordered_cores.extend(row)

                store = level_0.aszarr()
                z = zarr.open(store, mode='r')
                if isinstance(z, zarr.Group):
                    z = z['0'] if '0' in z else z[list(z.keys())[0]]

                is_channel_first = not (z.ndim == 3 and z.shape[0] > z.shape[2])

                center = (img_w // 2, img_h // 2)
                M_inv = cv2.getRotationMatrix2D(center, -detected_angle, 1.0)

                for idx, core in enumerate(ordered_cores):
                    core_id  = idx + 1
                    core_dir = os.path.join(STAGING_BASE, f"Core_{core_id:02d}")
                    os.makedirs(core_dir, exist_ok=True)

                    rotated_point = np.array([[core['cx'], core['cy']]], dtype=np.float32)
                    ones = np.ones(shape=(1, 1))
                    point_ones = np.hstack([rotated_point, ones])
                    original_point = M_inv.dot(point_ones.T).T
                    orig_cx, orig_cy = original_point[0][0], original_point[0][1]

                    high_cx, high_cy = int(orig_cx * scale_x), int(orig_cy * scale_y)
                    high_box  = int(low_res_box_size * scale_x)
                    half_box  = high_box // 2

                    start_x, start_y = max(0, high_cx - half_box), max(0, high_cy - half_box)
                    end_x, end_y     = min(w_high, high_cx + half_box), min(h_high, high_cy + half_box)

                    if is_channel_first:
                        crop = z[:, start_y:end_y, start_x:end_x]
                    else:
                        crop = z[start_y:end_y, start_x:end_x, :]
                        crop = np.transpose(crop, (2, 0, 1))

                    if detected_angle != 0.0:
                        crop = np.ascontiguousarray(crop, dtype=np.float32)
                        c, h, w = crop.shape
                        center_c = (w // 2, h // 2)
                        M_rot = cv2.getRotationMatrix2D(center_c, detected_angle, 1.0)
                        rotated_channels = []
                        for ch in range(c):
                            rotated_channels.append(cv2.warpAffine(
                                crop[ch, :, :], M_rot, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                            ))
                        crop = np.stack(rotated_channels, axis=0)

                    crop = np.clip(crop, 0, 65535)
                    crop = np.ascontiguousarray(crop, dtype=np.uint16)
                    c_dim, h_dim, w_dim = crop.shape

                    out_name = f"{tma_name}_Core{core_id:02d}.ome.tif"
                    out_full = os.path.join(core_dir, out_name)

                    metadata = {
                        'axes': 'CYX', 'Channel': {'Name': CHANNEL_NAMES},
                        'PhysicalSizeX': PIXEL_SIZE_UM, 'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeY': PIXEL_SIZE_UM, 'PhysicalSizeYUnit': 'µm',
                    }
                    tifffile.imwrite(out_full, crop, photometric='minisblack',
                                     metadata=metadata, compression=None)

                    thumb_raw = crop[0, :, :]
                    p99_thumb = np.percentile(thumb_raw, 99)
                    if p99_thumb == 0:
                        p99_thumb = thumb_raw.max() if thumb_raw.max() > 0 else 1
                    thumb_norm = np.clip((thumb_raw / p99_thumb) * 255.0, 0, 255).astype(np.uint8)
                    new_h = 512
                    new_w = int((w_dim / h_dim) * new_h)
                    thumb_resized = cv2.resize(thumb_norm, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(core_dir, f"{tma_name}_Core{core_id:02d}_thumb.jpg"), thumb_resized)

                logger.info(f"  Saved {len(ordered_cores)} cores to Core_XX/ folders")
                total_cores_extracted += len(ordered_cores)
                successful_slides += 1

        except Exception as e:
            logger.error(f" ERROR: {e}")
            failed_slides.append((tma_name, str(e)))
            import traceback
            traceback.print_exc()

    logger.info("─" * 60)
    logger.info(f"PHASE 1 COMPLETE — {successful_slides}/{len(config.TMA_FILES)} slides, "
               f"{total_cores_extracted} cores extracted")
    if failed_slides:
        for tma_name, reason in failed_slides:
            logger.warning(f"  Failed: {tma_name}: {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — CONFORM ALL CORES TO A CANONICAL SHAPE
# ─────────────────────────────────────────────────────────────────────────────

def canonical_shape(tif_paths: list) -> tuple:
    """Most common (H, W) across a core's slices; ties broken by larger shape."""
    shapes = [load_as_chw(p).shape[1:] for p in tif_paths]
    counts = Counter(shapes)
    ranked = sorted(counts.items(), key=lambda x: (x[1], x[0][0] * x[0][1]), reverse=True)
    best_shape, best_count = ranked[0]
    logger.info("  Shape distribution: " + ", ".join(f"{h}×{w}×{cnt}" for (h, w), cnt in ranked))
    logger.info(f"  Canonical shape: {best_shape[0]}×{best_shape[1]} "
               f"({best_count}/{len(tif_paths)} already match)")
    return best_shape


def conform_one_core(core_name: str, input_dir: str, output_dir: str) -> dict:
    tif_paths = sorted(glob.glob(os.path.join(input_dir, "*.ome.tif")))
    if not tif_paths:
        logger.warning(f"[{core_name}] No .ome.tif files found — skipping.")
        return {"core": core_name, "n_slices": 0, "n_conformed": 0, "n_copied": 0}

    logger.info(f"[{core_name}] {len(tif_paths)} slices found.")
    target_h, target_w = canonical_shape(tif_paths)

    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

    n_conformed = n_copied = 0
    for src_path in tif_paths:
        fname    = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, fname)
        arr = load_as_chw(src_path)
        h, w = arr.shape[1], arr.shape[2]

        if h == target_h and w == target_w:
            if not args.dry_run:
                shutil.copy2(src_path, dst_path)
            n_copied += 1
        else:
            logger.info(f"  {fname}: {h}×{w} -> {target_h}×{target_w} "
                       f"({'crop' if h > target_h or w > target_w else 'pad'})")
            if not args.dry_run:
                conformed = conform_slice(arr, target_h, target_w)
                tifffile.imwrite(dst_path, conformed, photometric='minisblack',
                                 metadata={'axes': 'CYX'}, compression='deflate',
                                 compressionargs={'level': 6})
            n_conformed += 1

    logger.info(f"[{core_name}] Done — {n_copied} copied, {n_conformed} conformed.")
    return {"core": core_name, "n_slices": len(tif_paths),
            "n_conformed": n_conformed, "n_copied": n_copied}


def conform_all_cores():
    logger.info("=" * 70)
    logger.info("PHASE 2 — CONFORM TO CANONICAL SHAPE")
    logger.info("=" * 70)
    if args.dry_run:
        logger.info("DRY RUN — no files will be written.")

    core_dirs = sorted(d for d in glob.glob(os.path.join(STAGING_BASE, "*")) if os.path.isdir(d))
    if not core_dirs:
        logger.error(f"No core folders found under {STAGING_BASE}")
        return

    results = []
    for core_dir in core_dirs:
        core_name  = os.path.basename(core_dir)
        output_dir = os.path.join(FINAL_BASE, core_name)
        results.append(conform_one_core(core_name, core_dir, output_dir))

    total_slices    = sum(r["n_slices"] for r in results)
    total_conformed = sum(r["n_conformed"] for r in results)
    total_copied    = sum(r["n_copied"] for r in results)
    logger.info("─" * 60)
    logger.info(f"PHASE 2 COMPLETE — {len(results)} cores, {total_slices} slices "
               f"({total_copied} copied, {total_conformed} conformed)")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — GENERATE TISSUE MASKS ON THE CONFORMED IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def generate_masks_for_conformed():
    logger.info("=" * 70)
    logger.info("PHASE 3 — TISSUE MASKS ON CONFORMED IMAGES")
    logger.info("=" * 70)
    if args.dry_run:
        logger.info("DRY RUN — no files will be written.")

    core_dirs = sorted(d for d in glob.glob(os.path.join(FINAL_BASE, "*")) if os.path.isdir(d))
    if not core_dirs:
        logger.error(f"No core folders found under {FINAL_BASE}")
        return

    n_masks = 0
    n_failed = 0
    for core_dir in core_dirs:
        core_name = os.path.basename(core_dir)
        tif_paths = sorted(glob.glob(os.path.join(core_dir, "*.ome.tif")))
        for tif_path in tif_paths:
            try:
                vol = load_as_chw(tif_path)
                mask = compute_tissue_mask_fullres(vol)
                mask_path = get_mask_sibling_path(tif_path)
                if not args.dry_run:
                    cv2.imwrite(mask_path, mask)
                n_masks += 1
            except Exception as exc:
                logger.error(f"[{core_name}] Mask failed for {os.path.basename(tif_path)}: {exc}")
                n_failed += 1
        logger.info(f"[{core_name}] {len(tif_paths)} mask(s) generated.")

    logger.info("─" * 60)
    logger.info(f"PHASE 3 COMPLETE — {n_masks} masks written, {n_failed} failed")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not args.skip_extract:
        extract_cores_from_slides()
    else:
        logger.info("Skipping phase 1 (extraction) — using existing STAGING_BASE contents.")

    if not args.skip_conform:
        conform_all_cores()
    else:
        logger.info("Skipping phase 2 (conforming) — using existing FINAL_BASE contents.")

    if not args.skip_mask:
        generate_masks_for_conformed()
    else:
        logger.info("Skipping phase 3 (mask generation).")

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Staging (raw crops):    {STAGING_BASE}")
    logger.info(f"  Final (conformed+mask): {FINAL_BASE}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()