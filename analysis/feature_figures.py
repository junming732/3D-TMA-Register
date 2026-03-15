"""
Feature Figure Generator
========================
Produces seven publication-quality figures illustrating the concepts in
the feature-based registration framework section:

  AKAZE figures:
    fig1_detectors_{ch}.png   — raw image | DoH response | keypoints
    fig2_anms_{ch}.png        — raw keypoints | after ANMS
    fig3_descriptor_{ch}.png  — patch | 3×3 grid | Hamming distance
    fig4_matching_{ch}.png    — all candidates | after ratio test | after RANSAC

  SIFT figures:
    fig5_dog_vs_doh_{ch}.png    — DoG vs DoH response + keypoints
    fig6_sift_desc_{ch}.png     — SIFT 128-D descriptor vs M-LDB binary

Usage (single line):
  python feature_figures.py --data_root /path/to/TMA_Cores_Grouped_Rotate_Conformed --core_name Core_11 --slice_a 0 --slice_b 1 --channel CK --out_dir ./thesis_figs

Arguments:
  --data_root   path to TMA_Cores_Grouped_Rotate_Conformed
  --core_name   core subfolder (e.g. Core_11)
  --slice_a     index of first slice  (default 0)
  --slice_b     index of second slice (default 1)  — used for matching figure
  --channel     int 0-7 OR name: DAPI CD31 GAP43 NFP CD3 CD163 CK AF
  --out_dir     output directory (default .)
  --n_anms      number of keypoints to keep after ANMS (default 300)

Requires: numpy matplotlib scipy tifffile opencv-python
"""

import os
import re
import glob
import argparse

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tifffile
from scipy.ndimage import gaussian_filter, maximum_filter

matplotlib.use('Agg')

DPI = 300
CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']

# ── colour palette (consistent across all figures) ───────────────────────────
C_PURPLE  = '#534AB7'
C_AMBER   = '#EF9F27'
C_TEAL    = '#1D9E75'
C_CORAL   = '#E85D24'
C_BLUE    = '#378ADD'
C_RED     = '#E24B4A'


# =============================================================================
# DATA LOADING  (mirrors registration script exactly)
# =============================================================================

def _sort_key(fname):
    m = re.search(r"TMA_(\d+)_", os.path.basename(fname))
    return int(m.group(1)) if m else 0


def load_slice(data_root, core_name, slice_idx):
    folder = os.path.join(data_root, core_name)
    files  = sorted(glob.glob(os.path.join(folder, "*.ome.tif")), key=_sort_key)
    if not files:
        raise FileNotFoundError(f"No .ome.tif in {folder}")
    if not (0 <= slice_idx < len(files)):
        raise IndexError(f"slice_idx={slice_idx} out of range 0..{len(files)-1}")
    arr = tifffile.imread(files[slice_idx])
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.moveaxis(arr, -1, 0)
    print(f"  Loaded   : {os.path.basename(files[slice_idx])}  shape={arr.shape}")
    return arr


def extract_channel(arr, channel):
    """Preprocess exactly as prepare_ck() in the registration script."""
    if isinstance(channel, str):
        idx = [n.upper() for n in CHANNEL_NAMES].index(channel.upper())
    else:
        idx = int(channel)
    ch_name = CHANNEL_NAMES[idx] if idx < len(CHANNEL_NAMES) else f"ch{idx}"
    raw     = arr[idx].astype(np.float32)
    log_img = np.log1p(raw)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm    = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return norm.astype(np.float64) / 255.0, ch_name


# =============================================================================
# SHARED HELPERS
# =============================================================================

def downsample(image, max_side=512):
    step = max(1, image.shape[0] // max_side)
    return image[::step, ::step], step


def compute_hessian(image):
    Lxx  = np.gradient(np.gradient(image, axis=1), axis=1)
    Lyy  = np.gradient(np.gradient(image, axis=0), axis=0)
    Lxy  = np.gradient(np.gradient(image, axis=1), axis=0)
    detH = Lxx * Lyy - Lxy ** 2
    return detH


def nms_keypoints(response, threshold_frac=0.10, min_dist=6, border=24):
    """
    Spatial non-maximum suppression.
    Returns list of (row, col, response_value).
    """
    thresh = response.max() * threshold_frac
    lm     = maximum_filter(response, size=min_dist * 2 + 1)
    mask   = (response == lm) & (response > thresh)
    mask[:border,  :]  = False
    mask[-border:, :]  = False
    mask[:,  :border]  = False
    mask[:, -border:]  = False
    r, c = np.where(mask)
    return list(zip(r.tolist(), c.tolist(), response[r, c].tolist()))


def anms(keypoints, n_keep):
    """
    Adaptive Non-Maximal Suppression (Brown et al. 2005).
    Retains n_keep keypoints with the largest minimum suppression radius.
    c = 0.9 robustness coefficient.
    """
    if len(keypoints) <= n_keep:
        return keypoints
    c    = 0.9
    pts  = np.array([(r, c_, v) for r, c_, v in keypoints])
    rs   = np.full(len(pts), np.inf)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if pts[j, 2] > c * pts[i, 2]:
                d = (pts[i, 0] - pts[j, 0])**2 + (pts[i, 1] - pts[j, 1])**2
                if d < rs[i]:
                    rs[i] = d
    idx = np.argsort(-rs)[:n_keep]
    return [keypoints[i] for i in idx]


def extract_patch(image, cy, cx, size=48):
    h  = size // 2
    r0 = max(0, cy - h);  r1 = min(image.shape[0], cy + h)
    c0 = max(0, cx - h);  c1 = min(image.shape[1], cx + h)
    p  = image[r0:r1, c0:c1]
    if p.shape != (size, size):
        pad = np.zeros((size, size), dtype=p.dtype)
        pad[:p.shape[0], :p.shape[1]] = p
        return pad
    return p.copy()


def cell_avg(patch, row, col, n=3):
    H, W = patch.shape
    ch, cw = H // n, W // n
    return patch[row*ch:(row+1)*ch, col*cw:(col+1)*cw].mean()


def mldb_bits(patch, n=3):
    avgs  = {(r, c): cell_avg(patch, r, c, n) for r in range(n) for c in range(n)}
    pairs = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n: pairs.append(((r, c), (r, c+1)))
            if r + 1 < n: pairs.append(((r, c), (r+1, c)))
            if r + 1 < n and c + 1 < n: pairs.append(((r, c), (r+1, c+1)))
    return [1 if avgs[a] > avgs[b] else 0 for a, b in pairs]


def hamming(b1, b2):
    return sum(a != b for a, b in zip(b1, b2))


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved : {path}")
    plt.close(fig)


# =============================================================================
# ANMS — exact copy of apply_anms() from registration script
# Uses cv2.KeyPoint.response, sorts by response, same c_robust logic
# =============================================================================

def apply_anms(keypoints, descriptors, num_to_keep=2000, c_robust=0.9,
               max_compute=5000):
    """
    Adaptive Non-Maximum Suppression.
    Mirrors apply_anms() in the registration script exactly:
      - sort by response descending, take top max_compute
      - for each point i, radius = min dist to any j where response[j]*c > response[i]
      - keep num_to_keep points with largest radii
    """
    n = len(keypoints)
    if n <= num_to_keep or descriptors is None:
        return keypoints, descriptors

    coords    = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])
    sort_idx  = np.argsort(responses)[::-1][:max_compute]
    n         = len(sort_idx)
    coords    = coords[sort_idx]
    responses = responses[sort_idx]
    radii     = np.full(n, np.inf)

    for i in range(1, n):
        stronger = responses[:i] * c_robust > responses[i]
        if np.any(stronger):
            diffs    = coords[:i][stronger] - coords[i]
            radii[i] = np.min(np.sum(diffs ** 2, axis=1))

    best  = np.argsort(radii)[::-1][:num_to_keep]
    final = sort_idx[best]
    return tuple(keypoints[i] for i in final), descriptors[final]


# =============================================================================
# FIGURE 1 — FEATURE DETECTORS
# Three panels: raw image | DoH heatmap | keypoints (OpenCV AKAZE, same as
# registration script: threshold=0.0003, full-res uint8)
# =============================================================================

def fig_detectors(image, ch_name, out_dir):
    print("\n[1/4] Feature Detectors figure")

    # ── run OpenCV AKAZE exactly as in registration script ────────────────────
    img_u8   = (image * 255).astype(np.uint8)
    detector = cv2.AKAZE_create(threshold=0.0003)   # AKAZE_THRESHOLD from reg script
    kp_raw, desc_raw = detector.detectAndCompute(img_u8, None)
    print(f"  Keypoints (OpenCV AKAZE, raw): {len(kp_raw)}")

    # ── DoH response map for visualisation (downsampled) ──────────────────────
    img_ds, step = downsample(image)
    print(f"  Computing DoH on downsampled image {img_ds.shape} ...")
    detH  = compute_hessian(img_ds)
    scale = 1.0 / step

    # ── subsample keypoints for drawing only — do not affect ANMS ─────────────
    # Drawing tens of thousands of circles hangs matplotlib.
    # Sort by response and take top 2000 for display only.
    kp_display = sorted(kp_raw, key=lambda k: k.response, reverse=True)[:2000]
    print(f"  Displaying top {len(kp_display)} keypoints by response (drawing cap)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'Feature Detection — Determinant of Hessian  |  Channel: {ch_name}',
        fontsize=13, y=1.01
    )

    # Panel A: raw image
    axes[0].imshow(img_ds, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('(a)  Input image\n(log-normalised)', fontsize=11)
    axes[0].axis('off')

    # Panel B: DoH response — clip colormap to ±99th percentile so tissue
    # structure is visible rather than drowned by the flat background
    vmax = np.percentile(np.abs(detH), 99)
    vmax = max(vmax, 1e-6)   # guard against all-zero
    im = axes[1].imshow(detH, cmap='RdBu_r', origin='upper',
                        vmin=-vmax, vmax=vmax)
    axes[1].set_title(
        r'(b)  $\det(\mathcal{H}) = L_{xx}L_{yy} - L_{xy}^2$'
        '\nred = blob  |  blue = saddle/edge  |  white = flat',
        fontsize=11
    )
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='response')

    # Panel C: top keypoints overlaid
    axes[2].imshow(img_ds, cmap='gray', vmin=0, vmax=1)
    max_r = max((kp.response for kp in kp_display), default=1.0) + 1e-9
    for kp in kp_display:
        x, y = kp.pt
        rad  = 2 + 10 * (kp.response / max_r)
        axes[2].add_patch(plt.Circle(
            (x * scale, y * scale), rad,
            fill=False, edgecolor=C_PURPLE, lw=0.8
        ))
    axes[2].set_title(
        f'(c)  Interest points: {len(kp_raw)} total\n'
        f'top {len(kp_display)} by response shown',
        fontsize=11
    )
    axes[2].axis('off')

    plt.tight_layout()
    save(fig, os.path.join(out_dir, f'fig1_detectors_{ch_name}.png'))
    return kp_raw, desc_raw, img_ds, scale


# =============================================================================
# FIGURE 2 — ANMS
# Two panels: raw AKAZE keypoints | after ANMS
# Uses apply_anms() identical to registration script (ANMS_KEEP=2000)
# =============================================================================

def fig_anms(img_ds, kp_raw, desc_raw, scale, ch_name, n_keep, out_dir):
    print("\n[2/4] ANMS figure")

    kp_anms, _ = apply_anms(kp_raw, desc_raw, num_to_keep=n_keep)
    print(f"  Before ANMS: {len(kp_raw)}   After ANMS: {len(kp_anms)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'Adaptive Non-Maximal Suppression (ANMS)  |  Channel: {ch_name}',
        fontsize=13, y=1.01
    )

    def draw_kps(ax, kps, title, max_draw=3000):
        ax.imshow(img_ds, cmap='gray', vmin=0, vmax=1)
        # subsample for display only — all points still passed to ANMS
        display = kps
        if len(kps) > max_draw:
            idx     = np.random.choice(len(kps), max_draw, replace=False)
            display = [kps[i] for i in idx]
        for kp in display:
            x, y = kp.pt
            ax.plot(x * scale, y * scale, '.',
                    color=C_AMBER, markersize=3, alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    draw_kps(axes[0], kp_raw,
             f'(a)  After NMS only\n'
             f'{len(kp_raw)} keypoints total  (3000 shown)\n'
             f'clustered in texture regions')
    draw_kps(axes[1], kp_anms,
             f'(b)  After ANMS  ($N={n_keep}$)\n'
             f'{len(kp_anms)} keypoints — spatially uniform')

    plt.tight_layout()
    save(fig, os.path.join(out_dir, f'fig2_anms_{ch_name}.png'))
    return kp_anms


# =============================================================================
# FIGURE 3 — FEATURE DESCRIPTOR
# Shows a real matched pair between fixed and moving slice:
#   (a) patch from fixed slice  (b) matched patch from moving slice
#   (c) 3×3 grid fixed          (d) 3×3 grid moving
#   (e) Hamming distance between the two descriptors
# =============================================================================

def fig_descriptor(image_a, image_b, ch_name, out_dir):
    print("\n[3/4] Feature Descriptor figure")

    def to_uint8(img): return (img * 255).astype(np.uint8)

    # ── run AKAZE on both slices, find a good matched pair ────────────────────
    akaze        = cv2.AKAZE_create(threshold=0.0003)
    kp_a, desc_a = akaze.detectAndCompute(to_uint8(image_a), None)
    kp_b, desc_b = akaze.detectAndCompute(to_uint8(image_b), None)

    if desc_a is None or desc_b is None or len(kp_a) == 0 or len(kp_b) == 0:
        print("  No descriptors — skipping.")
        return

    bf          = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(desc_a, desc_b, k=2)
    good        = [m for m, n in knn_matches
                   if len([m, n]) == 2 and m.distance < 0.8 * n.distance]

    if not good:
        print("  No good matches — skipping.")
        return

    # pick the match whose fixed-slice patch has the highest mean tissue content
    img_a_ds, step_a = downsample(image_a)
    img_b_ds, step_b = downsample(image_b)
    scale_a = 1.0 / step_a
    scale_b = 1.0 / step_b

    def patch_from_kp(img_ds, kp, scale, size=48):
        x, y = kp.pt
        return extract_patch(img_ds, int(y * scale), int(x * scale), size)

    best_match = max(
        good,
        key=lambda m: patch_from_kp(img_a_ds, kp_a[m.queryIdx], scale_a).mean()
    )

    kp1 = kp_a[best_match.queryIdx]
    kp2 = kp_b[best_match.trainIdx]
    p1  = patch_from_kp(img_a_ds, kp1, scale_a)
    p2  = patch_from_kp(img_b_ds, kp2, scale_b)
    bits1 = mldb_bits(p1, 3)
    bits2 = mldb_bits(p2, 3)
    hdist = hamming(bits1, bits2)
    cs    = 48 // 3
    n_bits = len(bits1)

    x1, y1 = int(kp1.pt[0]), int(kp1.pt[1])
    x2, y2 = int(kp2.pt[0]), int(kp2.pt[1])
    print(f"  Fixed  KP at ({x1},{y1})  Moving KP at ({x2},{y2})")
    print(f"  Hamming distance: {hdist} / {n_bits}  |  raw distance: {best_match.distance}")

    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12, 18))
    fig.suptitle(
        f'M-LDB Feature Descriptor — Matched Pair  |  Channel: {ch_name}',
        fontsize=13, y=1.01
    )
    gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.3,
                  height_ratios=[1, 1, 1.5, 0.35])

    ax_pa = fig.add_subplot(gs[0, 0])
    ax_pb = fig.add_subplot(gs[0, 1])
    ax_ga = fig.add_subplot(gs[1, 0])
    ax_gb = fig.add_subplot(gs[1, 1])
    ax_aa = fig.add_subplot(gs[2, 0])
    ax_ab = fig.add_subplot(gs[2, 1])
    ax_hd = fig.add_subplot(gs[3, :])

    def draw_grid(ax, patch, label, color):
        ax.imshow(patch, cmap='gray', vmin=0, vmax=1, alpha=0.6)
        for i in range(1, 3):
            ax.axhline(i * cs, color=color, lw=2.0)
            ax.axvline(i * cs, color=color, lw=2.0)
        for row in range(3):
            for col in range(3):
                v = cell_avg(patch, row, col, 3)
                ax.text(col*cs + cs//2, row*cs + cs//2, f'{v:.3f}',
                        ha='center', va='center', fontsize=11, color='white',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor=color,
                                  alpha=0.85, edgecolor='none'))
        ax.set_title(label, fontsize=11)
        ax.axis('off')

    def draw_arrows_with_bits(ax, patch, bits, label):
        ax.imshow(patch, cmap='gray', vmin=0, vmax=1, alpha=0.4)
        for i in range(1, 3):
            ax.axhline(i * cs, color='gray', lw=1.0, alpha=0.5)
            ax.axvline(i * cs, color='gray', lw=1.0, alpha=0.5)
        pairs = []
        for r in range(3):
            for c_ in range(3):
                if c_ + 1 < 3: pairs.append(((r, c_), (r, c_+1)))
                if r  + 1 < 3: pairs.append(((r, c_), (r+1, c_)))
                if r  + 1 < 3 and c_ + 1 < 3:
                    pairs.append(((r, c_), (r+1, c_+1)))
        for bit, (a, b) in zip(bits, pairs):
            src = (a[1]*cs + cs//2, a[0]*cs + cs//2)
            dst = (b[1]*cs + cs//2, b[0]*cs + cs//2)
            col = C_CORAL if bit == 1 else C_BLUE
            ax.annotate('', xy=dst, xytext=src,
                        arrowprops=dict(arrowstyle='->', lw=1.8, color=col))
        ax.legend(handles=[
            Line2D([0],[0], color=C_CORAL, lw=2, label='bit = 1  (A > B)'),
            Line2D([0],[0], color=C_BLUE,  lw=2, label='bit = 0  (A ≤ B)'),
        ], fontsize=9, loc='lower right')
        ax.set_title(label, fontsize=11)
        # bit string printed below the image using axes coordinates
        cols_per_line = 16
        for line_i, start in enumerate(range(0, len(bits), cols_per_line)):
            chunk = bits[start:start + cols_per_line]
            y_pos = -0.07 - line_i * 0.10
            for j, bv in enumerate(chunk):
                ax.text((j + 0.5) / cols_per_line, y_pos, str(bv),
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=8, fontweight='bold',
                        color=C_CORAL if bv == 1 else C_BLUE)
        ax.axis('off')

    # ── Row 0: patches ────────────────────────────────────────────────────────
    ax_pa.imshow(p1, cmap='gray', vmin=0, vmax=1)
    ax_pa.plot(24, 24, '+', color=C_CORAL, markersize=14, markeredgewidth=2.5)
    ax_pa.set_title(f'(a)  Fixed slice — patch at ({x1},{y1})', fontsize=11)
    ax_pa.axis('off')

    ax_pb.imshow(p2, cmap='gray', vmin=0, vmax=1)
    ax_pb.plot(24, 24, '+', color=C_TEAL, markersize=14, markeredgewidth=2.5)
    ax_pb.set_title(f'(b)  Moving slice — matched patch at ({x2},{y2})', fontsize=11)
    ax_pb.axis('off')

    # ── Row 1: 3×3 grids ─────────────────────────────────────────────────────
    draw_grid(ax_ga, p1,
              r'(c)  Fixed — 3×3 grid  $b_{ij}=\mathbf{1}[\bar{I}_i>\bar{I}_j]$',
              C_PURPLE)
    draw_grid(ax_gb, p2,
              r'(d)  Moving — 3×3 grid  $b_{ij}=\mathbf{1}[\bar{I}_i>\bar{I}_j]$',
              C_TEAL)

    # ── Row 2: arrows + bit strings below ────────────────────────────────────
    draw_arrows_with_bits(ax_aa, p1, bits1,
                          '(e)  Fixed — pair comparisons')
    draw_arrows_with_bits(ax_ab, p2, bits2,
                          '(f)  Moving — pair comparisons')

    # ── Row 3: Hamming distance badge ─────────────────────────────────────────
    ax_hd.axis('off')
    ax_hd.text(0.38, 0.6, 'Hamming distance:',
               ha='right', va='center', fontsize=12,
               color='gray', transform=ax_hd.transAxes)
    ax_hd.text(0.40, 0.6, f'{hdist} / {n_bits} bits',
               ha='left', va='center', fontsize=14, fontweight='bold',
               color=C_CORAL if hdist > n_bits // 2 else C_TEAL,
               transform=ax_hd.transAxes)
    ax_hd.text(0.5, 0.1,
               f'raw descriptor distance = {best_match.distance}  '
               f'|  accepted by Lowe ratio test  ($r=0.8$)',
               ha='center', va='center', fontsize=11,
               color='gray', transform=ax_hd.transAxes)

    plt.savefig(os.path.join(out_dir, f'fig3_descriptor_{ch_name}.png'),
                dpi=DPI, bbox_inches='tight')
    print(f"  Saved : {os.path.join(out_dir, f'fig3_descriptor_{ch_name}.png')}")
    plt.close(fig)


# =============================================================================
# FIGURE 4 — FEATURE MATCHING
# Three panels: all candidate matches | after ratio test | after RANSAC
# Uses two slices of the same core
# =============================================================================

def fig_matching(image_a, image_b, ch_name, out_dir):
    print("\n[4/4] Feature Matching figure")

    # ── run AKAZE detector + descriptor via OpenCV ────────────────────────────
    def to_uint8(img):
        return (img * 255).astype(np.uint8)

    akaze = cv2.AKAZE_create(threshold=0.0003)   # same as registration script
    kp_a, desc_a = akaze.detectAndCompute(to_uint8(image_a), None)
    kp_b, desc_b = akaze.detectAndCompute(to_uint8(image_b), None)
    print(f"  Slice A keypoints: {len(kp_a)}  Slice B: {len(kp_b)}")

    if desc_a is None or desc_b is None or len(kp_a) == 0 or len(kp_b) == 0:
        print("  No descriptors found — skipping matching figure.")
        return

    # ── brute-force Hamming matching ──────────────────────────────────────────
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_knn = bf.knnMatch(desc_a, desc_b, k=2)

    # all candidates (before ratio test)
    all_matches = [m for pair in matches_knn for m in pair[:1]]

    # after Lowe ratio test r = 0.8
    good_matches = []
    for pair in matches_knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
    print(f"  Candidate matches: {len(all_matches)}  After ratio test: {len(good_matches)}")

    # ── RANSAC homography to find inliers ─────────────────────────────────────
    inlier_matches = []
    if len(good_matches) >= 4:
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good_matches])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good_matches])
        _, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_matches = [m for m, keep in zip(good_matches, mask.ravel()) if keep]
    print(f"  After RANSAC: {len(inlier_matches)} inliers")

    # ── draw function ─────────────────────────────────────────────────────────
    def draw_matches(ax, img1, img2, kp1, kp2, matches, title, color, max_draw=80):
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        canvas  = np.zeros((max(h1, h2), w1 + w2), dtype=np.float64)
        canvas[:h1, :w1]   = img1
        canvas[:h2, w1:]   = img2
        ax.imshow(canvas, cmap='gray', vmin=0, vmax=1)
        # subsample for readability
        step = max(1, len(matches) // max_draw)
        for m in matches[::step]:
            x1, y1 = kp1[m.queryIdx].pt
            x2, y2 = kp2[m.trainIdx].pt
            ax.plot([x1, x2 + w1], [y1, y2],
                    '-', color=color, lw=0.6, alpha=0.6)
            ax.plot(x1,      y1, '.', color=color, markersize=3)
            ax.plot(x2 + w1, y2, '.', color=color, markersize=3)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    img_a_ds, _ = downsample(image_a)
    img_b_ds, _ = downsample(image_b)
    scale        = img_a_ds.shape[0] / image_a.shape[0]

    def scale_kp(kps):
        scaled = []
        for k in kps:
            x, y = k.pt
            sk   = cv2.KeyPoint(x * scale, y * scale, k.size * scale)
            scaled.append(sk)
        return scaled

    kp_a_ds = scale_kp(kp_a)
    kp_b_ds = scale_kp(kp_b)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Feature Matching  |  Channel: {ch_name}  '
        f'|  Slice A (left) vs Slice B (right)',
        fontsize=13, y=1.01
    )

    draw_matches(axes[0], img_a_ds, img_b_ds, kp_a_ds, kp_b_ds,
                 all_matches,
                 f'(a)  All candidate matches\n$|\\mathcal{{M}}_{{\\mathrm{{raw}}}}|$ = {len(all_matches)}',
                 color=C_BLUE)

    draw_matches(axes[1], img_a_ds, img_b_ds, kp_a_ds, kp_b_ds,
                 good_matches,
                 f'(b)  After Lowe ratio test  ($r=0.8$)\n$|\\mathcal{{M}}|$ = {len(good_matches)}',
                 color=C_TEAL)

    draw_matches(axes[2], img_a_ds, img_b_ds, kp_a_ds, kp_b_ds,
                 inlier_matches,
                 f'(c)  After RANSAC — inliers\n$|\\mathcal{{I}}|$ = {len(inlier_matches)}',
                 color=C_AMBER)

    plt.tight_layout()
    save(fig, os.path.join(out_dir, f'fig4_matching_{ch_name}.png'))


# =============================================================================
# FIGURE 5 — DoG vs DoH
# Left block: Gaussian scale space (4 σ levels) + DoG response
# Right block: DoH response
# Shows why the two detectors differ
# =============================================================================

def fig_dog_vs_doh(image, ch_name, out_dir):
    print("\n[5/6] DoG vs DoH figures (saved as two separate files)")

    def to_uint8(img): return (img * 255).astype(np.uint8)

    img_ds, step = downsample(image)
    scale        = 1.0 / step

    # ── DoG response (illustrative, single scale pair) ────────────────────────
    L1  = gaussian_filter(img_ds, sigma=1.0)
    L2  = gaussian_filter(img_ds, sigma=1.4)
    dog = L2 - L1
    kps_dog = nms_keypoints(np.abs(dog), threshold_frac=0.10)

    # ── DoH response (illustrative, single scale) ─────────────────────────────
    detH = compute_hessian(img_ds)

    # ── OpenCV AKAZE keypoints — same detector as rest of pipeline ────────────
    akaze      = cv2.AKAZE_create(threshold=0.0003)
    kp_akaze, _ = akaze.detectAndCompute(to_uint8(image), None)
    print(f"  DoG keypoints (illustrative): {len(kps_dog)}")
    print(f"  AKAZE keypoints (OpenCV):     {len(kp_akaze)}")

    def sym_vlim(arr, pct=99):
        return max(np.percentile(np.abs(arr), pct), 1e-6)

    # ── Figure 5a: DoG response only ─────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    fig.suptitle(
        f'DoG Detector Response (SIFT)  |  Channel: {ch_name}',
        fontsize=13
    )
    vdog = sym_vlim(dog)
    im0  = ax.imshow(dog, cmap='RdBu_r', origin='upper', vmin=-vdog, vmax=vdog)
    ax.set_title(
        r'$D(x,y,\sigma) = L(x,y,k\sigma) - L(x,y,\sigma)$'
        '\nred = positive  |  blue = negative  |  white = flat',
        fontsize=11
    )
    ax.axis('off')
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f'fig5a_dog_{ch_name}.png'))

    # ── Figure 5b: DoH response only ─────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    fig.suptitle(
        f'DoH Detector Response (AKAZE)  |  Channel: {ch_name}',
        fontsize=13
    )
    vdoh = sym_vlim(detH)
    im1  = ax.imshow(detH, cmap='RdBu_r', origin='upper', vmin=-vdoh, vmax=vdoh)
    ax.set_title(
        r'$\det(\mathcal{H}) = L_{xx}L_{yy} - L_{xy}^2$'
        '\nred = blob  |  blue = saddle/edge  |  white = flat',
        fontsize=11
    )
    ax.axis('off')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save(fig, os.path.join(out_dir, f'fig5b_doh_{ch_name}.png'))





# =============================================================================
# FIGURE 7 — SIFT 128-D descriptor vs M-LDB binary descriptor
# Left: gradient orientation histogram for one patch (SIFT)
# Right: binary bit grid for the same patch (M-LDB / AKAZE)
# =============================================================================

def fig_sift_desc_vs_mldb(image_a, image_b, ch_name, out_dir):
    print("\n[6/6] SIFT descriptor figure")

    def to_uint8(img): return (img * 255).astype(np.uint8)

    # ── find a good keypoint on the fixed slice via AKAZE ─────────────────────
    akaze        = cv2.AKAZE_create(threshold=0.0003)
    kp_a, _      = akaze.detectAndCompute(to_uint8(image_a), None)
    if not kp_a:
        print("  No keypoints — skipping.")
        return

    img_a_ds, step_a = downsample(image_a)
    sa = 1.0 / step_a

    def get_patch(kp, size=48):
        x, y = kp.pt
        return extract_patch(img_a_ds, int(y * sa), int(x * sa), size)

    # pick keypoint with highest mean tissue content
    best  = max(kp_a, key=lambda k: get_patch(k).mean())
    patch = get_patch(best)
    x1, y1 = int(best.pt[0]), int(best.pt[1])
    print(f"  SIFT descriptor patch at ({x1},{y1})")

    # ── SIFT-style 128-D gradient histogram ───────────────────────────────────
    gx        = np.gradient(patch, axis=1)
    gy        = np.gradient(patch, axis=0)
    mag       = np.sqrt(gx**2 + gy**2)
    ori       = np.degrees(np.arctan2(gy, gx)) % 360.0
    cell_size = 48 // 4
    edges     = np.linspace(0, 360, 9)
    grid      = np.zeros((4, 4, 8))
    for gr in range(4):
        for gc in range(4):
            r0, r1 = gr*cell_size, (gr+1)*cell_size
            c0, c1 = gc*cell_size, (gc+1)*cell_size
            grid[gr, gc], _ = np.histogram(
                ori[r0:r1, c0:c1].ravel(), bins=edges,
                weights=mag[r0:r1, c0:c1].ravel()
            )
    sift_vec = grid.ravel()   # 128-D

    from matplotlib.lines import Line2D

    # ── layout: 1 row × 2 panels ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'SIFT 128-D Descriptor  |  Channel: {ch_name}  |  Patch at ({x1},{y1})',
        fontsize=13, y=1.01
    )

    # Panel (a): patch with 4×4 SIFT grid overlaid
    axes[0].imshow(patch, cmap='gray', vmin=0, vmax=1)
    axes[0].plot(24, 24, '+', color=C_CORAL, markersize=14, markeredgewidth=2.5)
    for i in range(1, 4):
        axes[0].axhline(i * cell_size, color=C_BLUE, lw=1.2, alpha=0.8)
        axes[0].axvline(i * cell_size, color=C_BLUE, lw=1.2, alpha=0.8)
    axes[0].legend(handles=[
        Line2D([0],[0], color=C_BLUE, lw=1.5, label='SIFT 4×4 grid'),
    ], fontsize=10, loc='lower right')
    axes[0].set_title(
        '(a)  Patch around interest point\n4×4 cell grid',
        fontsize=11
    )
    axes[0].axis('off')

    # Panel (b): 128-D bar chart
    axes[1].bar(np.arange(128), sift_vec, color=C_BLUE, alpha=0.8, width=1.0)
    for i in range(1, 16):
        axes[1].axvline(i * 8 - 0.5, color='gray', lw=0.5, alpha=0.4)
    axes[1].set_xlabel('descriptor dimension  (16 cells × 8 bins)', fontsize=11)
    axes[1].set_ylabel('weighted gradient count', fontsize=11)
    axes[1].set_title(
        r'(b)  SIFT descriptor  $\mathbf{f} \in \mathbb{R}^{128}$'
        '\ngradient orientation histograms',
        fontsize=11
    )
    axes[1].set_xlim(-1, 128)
    axes[1].grid(True, alpha=0.15)

    plt.tight_layout()
    save(fig, os.path.join(out_dir, f'fig6_sift_desc_{ch_name}.png'))


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Feature figures — feature-based registration framework'
    )
    p.add_argument('--data_root', required=True,
                   help='Path to TMA_Cores_Grouped_Rotate_Conformed')
    p.add_argument('--core_name', required=True,
                   help='Core subfolder, e.g. Core_11')
    p.add_argument('--slice_a',  type=int, default=0,
                   help='First slice index (default 0)')
    p.add_argument('--slice_b',  type=int, default=1,
                   help='Second slice index for matching figure (default 1)')
    p.add_argument('--channel',  default='CK',
                   help='Channel name or index (default CK)')
    p.add_argument('--out_dir',  default='.',
                   help='Output directory (default .)')
    p.add_argument('--n_anms',   type=int, default=2000,
                   help='Keypoints to keep after ANMS (default 2000, matches registration script)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 60)
    print('Feature Figure Generator')
    print('=' * 60)
    print(f'  data_root : {args.data_root}')
    print(f'  core_name : {args.core_name}')
    print(f'  slice_a   : {args.slice_a}')
    print(f'  slice_b   : {args.slice_b}')
    print(f'  channel   : {args.channel}')
    print(f'  n_anms    : {args.n_anms}')
    print(f'  out_dir   : {args.out_dir}')

    # load slice A
    arr_a         = load_slice(args.data_root, args.core_name, args.slice_a)
    image_a, ch   = extract_channel(arr_a, args.channel)

    # load slice B (for matching figure)
    arr_b         = load_slice(args.data_root, args.core_name, args.slice_b)
    image_b, _    = extract_channel(arr_b, args.channel)

    # Fig 1 — detectors
    kp_raw, desc_raw, img_ds, scale = fig_detectors(image_a, ch, args.out_dir)

    # Fig 2 — ANMS  (n_anms default=2000 matches registration script)
    kp_anms = fig_anms(img_ds, kp_raw, desc_raw, scale, ch, args.n_anms, args.out_dir)

    # Fig 3 — descriptor (uses both slices to show a real matched pair)
    fig_descriptor(image_a, image_b, ch, args.out_dir)

    # Fig 4 — matching (uses full-res images, OpenCV AKAZE)
    fig_matching(image_a, image_b, ch, args.out_dir)

    # Fig 5 — DoG vs DoH
    fig_dog_vs_doh(image_a, ch, args.out_dir)

    # Fig 6 — SIFT 128-D descriptor (fixed slice only)
    fig_sift_desc_vs_mldb(image_a, image_b, ch, args.out_dir)

    print('\nDone. Figures saved to:', args.out_dir)
    print('  fig1_detectors, fig2_anms, fig3_descriptor, fig4_matching,')
    print('  fig5a_dog, fig5b_doh, fig6_sift_desc')