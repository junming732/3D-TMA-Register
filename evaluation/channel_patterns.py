"""
channel_patterns.py
────────────────────────────────────────────────────────────────────────────
Shared channel-preprocessing recipes, factored out of
akaze_romav2_multi_channel_warp_new.py so they can be reused by accuracy /
QC scripts (e.g. accuracy_landmarks_deform.py) WITHOUT importing the
registration script itself — that script calls `args = parser.parse_args()`
at module level, so importing it directly would try to parse whatever
script imported it and crash.

Keep this file's implementations byte-for-byte identical to the
corresponding functions in the registration script. If you change a recipe
there, change it here too (or better: make the registration script import
from here instead of defining its own copies).

Every pattern function takes single-channel float/uint arrays and returns
uint8 output — same contract as the registration script.
"""

import numpy as np
import cv2

# ─── Channel layout — mirrors akaze_romav2_multi_channel_warp_new.py ─────────
DAPI_CHANNEL_IDX = 0
CK_CHANNEL_IDX   = 6
AF_CHANNEL_IDX   = 7
CHANNEL_NAMES    = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

COLOR_LUT = {
    0: (0,   128, 255),
    1: (51,  255,  51),
    2: (255,  51,  51),
    3: (0,   255, 255),
    4: (255,   0, 255),
    5: (255, 255,   0),
    6: (255, 128,   0),
}


def prepare_ck(img_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (norm_lin, norm_log) — both uint8. See registration script."""
    img_float = img_arr.astype(np.float32)

    p_lo_lin, p_hi_lin = np.percentile(img_float[::4, ::4], (0.1, 99.9))
    norm_lin = cv2.normalize(
        np.clip(img_float, p_lo_lin, p_hi_lin), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    log_img    = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log   = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return norm_lin, norm_log


def clahe_normalize(img_arr: np.ndarray) -> np.ndarray:
    """VALIS-style adaptive-histogram-equalisation normalisation."""
    from skimage import exposure
    img_float = img_arr.astype(np.float32)
    img01 = exposure.rescale_intensity(img_float, out_range=(0, 1))
    eq    = exposure.equalize_adapthist(img01)
    return exposure.rescale_intensity(eq, out_range=(0, 255)).astype(np.uint8)


def _prepare_single(img_arr, lo=0.1, hi=99.5):
    img_float = img_arr.astype(np.float32)
    log_img   = np.log1p(img_float)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (lo, hi))
    return cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)


def prepare_3ch_fusion_from_channels(dapi_img, af_img, ck_img) -> np.ndarray:
    """(H, W, 3) uint8 RGB: DAPI->R, AF->G, CK->B, each log-normalised."""
    return np.stack([
        _prepare_single(dapi_img.astype(np.float32)),
        _prepare_single(af_img.astype(np.float32)),
        _prepare_single(ck_img.astype(np.float32)),
    ], axis=-1)


def prepare_color_lut_fusion_from_channels(channel_imgs: dict) -> np.ndarray:
    """
    (H, W, 3) uint8 RGB weighted-average color-LUT composite.
    channel_imgs: {channel_idx (0-6): raw image array}
    """
    any_img = next(iter(channel_imgs.values()))
    h, w = any_img.shape
    acc  = np.zeros((h, w, 3), dtype=np.float32)
    n    = len(COLOR_LUT)
    for idx, color in COLOR_LUT.items():
        norm      = _prepare_single(channel_imgs[idx].astype(np.float32)).astype(np.float32) / 255.0
        color_arr = np.array(color, dtype=np.float32) / 255.0
        acc      += norm[..., None] * color_arr[None, None, :]
    return np.clip(acc / n * 255.0, 0, 255).astype(np.uint8)


def rgb_to_gray_luma(rgb_img: np.ndarray) -> np.ndarray:
    """Collapse an (H,W,3) uint8 RGB composite to (H,W) uint8 via cv2 luma weights."""
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)


def load_slice_channel_raw_from_vol(vol, slice_idx, channel_idx, slice_idx_to_vol_z):
    """
    Load a single channel from the registered volume WITHOUT the 2nd/98th
    percentile [0,1] contrast stretch that
    landmark_accuracy_deform_common.load_slice_channel_from_vol applies.

    prepare_ck / clahe_normalize / prepare_color_lut_fusion / prepare_3ch_fusion
    all do their own log1p + percentile normalisation on RAW intensities —
    identical to what they receive in the registration script, where they're
    called directly on the raw multichannel volume. Feeding them the
    already-[0,1]-stretched output of load_slice_channel_from_vol would
    double-stretch and silently diverge from the registration script's actual
    output, so every pattern other than the original 'dapi'/'ck' rows must
    use this raw loader instead.
    """
    vol_z = slice_idx_to_vol_z.get(slice_idx)
    if vol_z is None:
        return None
    try:
        return vol[vol_z, channel_idx].astype(np.float32)
    except Exception:
        return None


# ─── Pattern registry ─────────────────────────────────────────────────────
# Each entry declares which raw channel indices it needs loaded per slice,
# a builder that turns {channel_idx: raw_img} into a single uint8 (H, W)
# grayscale image, a display label, and a default crop half-width for the
# adjacent-slice overlay plot.
#
# 'dapi' / 'ck' reproduce the ORIGINAL accuracy-script behaviour exactly:
# loader='stretched' (landmark_accuracy_deform_common.load_slice_channel_from_vol,
# its 2nd/98th-pct [0,1] stretch), raw crop passed straight to make_two_channel_rgb.
#
# Every other pattern needs loader='raw' (load_slice_channel_raw_from_vol above)
# so its builder's own log1p/percentile normalisation matches what the
# registration script actually computed — see load_slice_channel_raw_from_vol's
# docstring for why 'stretched' input would silently diverge.

PATTERN_REGISTRY = {
    'dapi': dict(
        channels = [DAPI_CHANNEL_IDX],
        loader   = 'stretched',
        builder  = lambda ch: ch[DAPI_CHANNEL_IDX],
        label    = 'DAPI',
        crop_half = 50,
    ),
    'ck': dict(
        channels = [CK_CHANNEL_IDX],
        loader   = 'stretched',
        builder  = lambda ch: ch[CK_CHANNEL_IDX],
        label    = 'CK',
        crop_half = 150,
    ),
    'dapi_clahe': dict(
        channels = [DAPI_CHANNEL_IDX],
        loader   = 'raw',
        builder  = lambda ch: clahe_normalize(ch[DAPI_CHANNEL_IDX]),
        label    = 'DAPI+CLAHE',
        crop_half = 50,
    ),
    'ck_clahe': dict(
        channels = [CK_CHANNEL_IDX],
        loader   = 'raw',
        builder  = lambda ch: clahe_normalize(ch[CK_CHANNEL_IDX]),
        label    = 'CK+CLAHE',
        crop_half = 150,
    ),
    'color_lut': dict(
        channels = list(COLOR_LUT.keys()),
        loader   = 'raw',
        builder  = lambda ch: rgb_to_gray_luma(prepare_color_lut_fusion_from_channels(ch)),
        label    = 'Color-LUT',
        crop_half = 150,
    ),
    '3ch_fusion': dict(
        channels = [DAPI_CHANNEL_IDX, AF_CHANNEL_IDX, CK_CHANNEL_IDX],
        loader   = 'raw',
        builder  = lambda ch: rgb_to_gray_luma(
            prepare_3ch_fusion_from_channels(ch[DAPI_CHANNEL_IDX], ch[AF_CHANNEL_IDX], ch[CK_CHANNEL_IDX])
        ),
        label    = 'DAPI+AF+CK fusion',
        crop_half = 150,
    ),
}