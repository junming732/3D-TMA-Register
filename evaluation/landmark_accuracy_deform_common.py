"""
landmark_accuracy_deform_common.py
====================================
Helpers shared specifically by the two landmark-accuracy scripts that
evaluate a saved-deformation-map (.npz) based registration pipeline:
accuracy_landmarks_bspline.py and accuracy_landmarks_roma.py.

NOT used by valis_accuracy_landmarks.py — VALIS stores its transform in a
pickled Registrar object and warps points via slide_obj.warp_xy() instead,
so it has no equivalent of find_deform_npz / warp_point.

These functions were previously duplicated identically (modulo comment
wording — including one script's docstring for make_two_channel_rgb-style
drift, since fixed) across both scripts. Extracted here so a real fix only
needs to happen once instead of being copied into two places and risking
the two copies silently disagreeing.
"""

import os
import glob
import logging
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_slice_filter(yaml_path, core_name):
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh) or {}
    raw = data.get(core_name)
    if raw is None:
        return None
    allowed = set()
    for part in str(raw).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            allowed.update(range(int(lo.strip()), int(hi.strip()) + 1))
        else:
            allowed.add(int(part))
    return allowed


def find_deform_npz(slice_idx, target_core, deform_folder):
    """Locate the saved deformation .npz for a given slice index."""
    for pat in [f"{target_core}_Z{slice_idx:03d}_*_deformation.npz",
                f"{target_core}_Z{slice_idx}_*_deformation.npz"]:
        m = glob.glob(os.path.join(deform_folder, pat))
        if m:
            return m[0]
    return None


def warp_point(x, y, npz_path):
    """
    Forward-warp (x, y) from original moving-slice space → registered space.

    The saved deformation is a two-step pipeline (mirrors register_slice):
      Step 1 — affine:  M_affine maps moving → fixed (applied directly to the point).
      Step 2 — remap:   map_x/map_y are BACKWARD maps (output→source in affine space).
                        We search for the output pixel whose source coordinate is
                        closest to (ax, ay) — the affine-transformed point.

    M_affine convention:
      cv2.warpAffine(image, M_affine, ...) uses M as output→source for images,
      which means M_affine @ [x, y, 1] transforms a point moving → fixed.
      Do NOT invert it for point-space transformation.

    Returns (wx, wy) in registered space.
    """
    d        = np.load(npz_path)
    M_affine = d['M_affine'].astype(np.float64)   # (2, 3)
    map_x    = d['map_x'].astype(np.float32)       # (H, W)
    map_y    = d['map_y'].astype(np.float32)       # (H, W)
    H, W     = map_x.shape

    # Step 1: apply affine directly (moving → fixed).
    pt     = np.array([x, y, 1.0], dtype=np.float64)
    xy_aff = M_affine @ pt
    ax, ay = float(xy_aff[0]), float(xy_aff[1])

    # Step 2: NN search — find output pixel (wx, wy) whose backward-map
    # source coordinate is closest to (ax, ay).
    search_r = 150
    x0 = max(0, int(ax) - search_r);  x1 = min(W, int(ax) + search_r)
    y0 = max(0, int(ay) - search_r);  y1 = min(H, int(ay) + search_r)
    patch_x = map_x[y0:y1, x0:x1]
    patch_y = map_y[y0:y1, x0:x1]
    dist2   = (patch_x - ax)**2 + (patch_y - ay)**2
    ry, rx  = np.unravel_index(np.argmin(dist2), dist2.shape)
    return float(x0 + rx), float(y0 + ry)


def load_slice_channel_from_vol(vol, slice_idx, channel_idx, slice_idx_to_vol_z):
    """
    Load and contrast-stretch a single channel from the registered volume.

    Uses slice_idx_to_vol_z to map TMA slice_idx to the correct Z position
    in the registered volume (Z, C, H, W). Loading from the registered volume
    (rather than the raw unregistered file list) ensures image content is
    aligned with the warped landmark coordinates.
    """
    vol_z = slice_idx_to_vol_z.get(slice_idx)
    if vol_z is None:
        logger.warning(f"slice_idx={slice_idx} not found in filtered slice list — skipping.")
        return None
    try:
        ch = vol[vol_z, channel_idx].astype(np.float32)
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 > p2:
            ch = np.clip((ch - p2) / (p98 - p2), 0, 1)
        return ch
    except Exception as e:
        logger.warning(f"Could not load vol_z={vol_z} ch={channel_idx}: {e}")
        return None