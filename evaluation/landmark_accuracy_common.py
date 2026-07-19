"""
landmark_accuracy_common.py
============================
Helpers shared by all three landmark-accuracy scripts
(accuracy_landmarks_bspline.py, accuracy_landmarks_roma.py,
valis_accuracy_landmarks.py), regardless of which registration pipeline
produced the transform being evaluated.

These four functions were previously duplicated identically (modulo comment
wording) across all three scripts. Extracted here so a real fix only needs
to happen once instead of being copied into three places and risking drift.
"""

import os
import re
import numpy as np


def z_json_to_slice_idx(z_json):
    """Convert a 1-based annotation Z index (as stored in the annotation JSON) to
    the 0-based slice index used throughout the registration pipeline."""
    return z_json - 1


def get_slice_number(filename):
    """Extract the TMA slice number from a filename like '...TMA_007_...'."""
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def make_two_channel_rgb(img_a, img_b):
    """
    Blend two greyscale images into a red (B = upper z) / green (A = lower z) overlay.
    Yellow pixels indicate agreement between the two slices.
    """
    h = max(img_a.shape[0] if img_a is not None else 0,
            img_b.shape[0] if img_b is not None else 0)
    w = max(img_a.shape[1] if img_a is not None else 0,
            img_b.shape[1] if img_b is not None else 0)
    a = img_a if img_a is not None else np.zeros((h, w), np.float32)
    b = img_b if img_b is not None else np.zeros((h, w), np.float32)
    if a.shape != b.shape:
        from skimage.transform import resize as sk_resize
        b = sk_resize(b, a.shape, anti_aliasing=True).astype(np.float32)
    r  = np.clip(b, 0, 1)   # red   — upper slice
    g  = np.clip(a, 0, 1)   # green — lower slice
    bl = np.zeros_like(r)   # no blue: overlap → yellow, single → red or green
    return np.stack([r, g, bl], axis=2)


def _annotate_overlay_ax(ax, rgb, wx_a, wy_a, wx_b, wy_b,
                         x0, x1, y0, y1, z_a, z_b,
                         dist_px, dist_um, row_label):
    """Shared helper: imshow + markers + arrow + label for one overlay panel."""
    mid_x = (wx_a + wx_b) / 2
    mid_y = (wy_a + wy_b) / 2

    ax.imshow(rgb, origin='upper', extent=[x0, x1, y1, y0])

    ax.scatter(wx_a, wy_a, c='#00ff00', s=180, zorder=5,
               edgecolors='white', linewidths=1.2, label=f"slice {z_a}")
    ax.scatter(wx_b, wy_b, c='#ff0000', s=180, zorder=5,
               edgecolors='white', linewidths=1.2, label=f"slice {z_b}")

    ax.annotate('', xy=(wx_b, wy_b), xytext=(wx_a, wy_a),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5))

    ax.text(mid_x, mid_y - 5,
            f"{dist_px:.1f} px / {dist_um:.1f} µm",
            ha='center', va='bottom', fontsize=14,
            color='yellow', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55, lw=0))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)

    ax.set_title(f"{row_label}  |  slice {z_a} → slice {z_b}  |  Δ = {dist_um:.1f} µm",
                 fontsize=24, pad=10)
    ax.set_xlabel("x (px)", fontsize=24)
    ax.set_ylabel("y (px)", fontsize=24)
    ax.tick_params(axis='both', labelsize=24)
    ax.legend(fontsize=24, loc='lower right',
              facecolor='black', labelcolor='white', framealpha=0.6)