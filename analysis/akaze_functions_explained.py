"""
AKAZE Core Functions — Visual Explainer
========================================
Reads a real slice from TMA_Cores_Grouped_Rotate_Conformed and lets you
choose the channel.  Covers three key concepts:

  1. Conductance function  c = g₂(|∇Lσ|)   [AKAZE paper Eq. 2 & 3]
  2. Determinant of Hessian (DoH)            [blob detector]
  3. M-LDB descriptor                        [binary patch encoding]

Usage:
  python akaze_functions_explained.py \
      --data_root /path/to/TMA_Cores_Grouped_Rotate_Conformed \
      --core_name CORE_001 \
      --slice_idx 0 \
      --channel CK

  --channel  int index (0-7) OR name: DAPI CD31 GAP43 NFP CD3 CD163 CK AF
  --slice_idx  which file in the sorted TMA_<N>_ list (default 0 = first slice)

Requires: numpy matplotlib scipy tifffile
Install:  pip install numpy matplotlib scipy tifffile
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import gaussian_filter

matplotlib.use('Agg')

# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL REGISTRY  (matches CHANNEL_NAMES in your registration script)
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_NAMES = ['DAPI', 'CD31', 'GAP43', 'NFP', 'CD3', 'CD163', 'CK', 'AF']


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING  — mirrors your registration script exactly
# ─────────────────────────────────────────────────────────────────────────────

def get_slice_number(filename):
    """Sort key: extract TMA_<number>_ from filename — same as your script."""
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def load_slice(data_root: str, core_name: str, slice_idx: int):
    """
    Load one .ome.tif from TMA_Cores_Grouped_Rotate_Conformed/<core_name>/.
    Returns (arr, filepath) where arr is (C, H, W) uint16.
    slice_idx indexes into the TMA_<N> sorted file list.
    """
    folder    = os.path.join(data_root, core_name)
    raw_files = glob.glob(os.path.join(folder, "*.ome.tif"))
    if not raw_files:
        raise FileNotFoundError(f"No .ome.tif files in: {folder}")

    file_list = sorted(raw_files, key=get_slice_number)
    n         = len(file_list)

    if not (0 <= slice_idx < n):
        raise IndexError(f"slice_idx={slice_idx} out of range 0..{n-1}")

    chosen = file_list[slice_idx]
    print(f"  File     : {os.path.basename(chosen)}  ({slice_idx}/{n-1})")

    arr = tifffile.imread(chosen)

    # Normalise axis order to (C, H, W) — same logic as your script
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.moveaxis(arr, -1, 0)

    print(f"  Shape    : {arr.shape}  dtype={arr.dtype}")
    return arr, chosen


def extract_channel(arr: np.ndarray, channel):
    """
    Pull one channel from (C, H, W) and preprocess it exactly as prepare_ck()
    does in the registration script — this is the image AKAZE actually sees.

    Steps (mirroring prepare_ck verbatim):
      1. Cast to float32
      2. log1p                              (boosts weak signal, matches your pipeline)
      3. Percentile clip  (0.1, 99.9) on   (sampled at [::4,::4] for speed)
      4. cv2.normalize → uint8  [0, 255]   (same as AKAZE input in registration)
      5. Return as float64 in [0, 1]        (divide by 255, for downstream maths)

    Returns (image float64 [0,1], channel_name string).
    """
    import cv2

    if isinstance(channel, str):
        names_up = [n.upper() for n in CHANNEL_NAMES]
        key      = channel.upper()
        if key not in names_up:
            raise ValueError(f"Channel '{channel}' not in {CHANNEL_NAMES}")
        ch_idx = names_up.index(key)
    else:
        ch_idx = int(channel)

    if ch_idx >= arr.shape[0]:
        raise IndexError(f"ch_idx={ch_idx} but array has {arr.shape[0]} channels")

    ch_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"ch{ch_idx}"
    print(f"  Channel  : index {ch_idx} → {ch_name}")

    raw = arr[ch_idx].astype(np.float32)

    # ── exact prepare_ck logic ────────────────────────────────────────────────
    log_img    = np.log1p(raw)
    p_lo, p_hi = np.percentile(log_img[::4, ::4], (0.1, 99.9))
    norm_log   = cv2.normalize(
        np.clip(log_img, p_lo, p_hi), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    # ─────────────────────────────────────────────────────────────────────────

    print(f"  Pre-proc : log1p → percentile clip ({p_lo:.3f}, {p_hi:.3f}) → uint8")
    print(f"  uint8 range: [{norm_log.min()}, {norm_log.max()}]")

    # Convert to float64 [0,1] for the maths in sections 1-3
    return norm_log.astype(np.float64) / 255.0, ch_name


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONDUCTANCE FUNCTION  c = g₂(|∇Lσ|)   [paper Eq. 2 & 3]
# ─────────────────────────────────────────────────────────────────────────────
#
# PDE (Eq. 1):    ∂L/∂t = div( c(x,y,t) · ∇L )
# Conductance (Eq. 2):  c = g( |∇Lσ| )    ← gradient of SMOOTHED image
# AKAZE uses g₂ (Eq. 3):
#
#   g₂ = 1 / ( 1 + |∇Lσ|² / λ² )
#
# λ = 70th percentile of |∇Lσ| histogram  (auto, no manual tuning)
# g₂(0)=1  g₂(λ)=0.5  g₂(∞)→0

def g2(s, lam):
    """AKAZE conductance g₂ — paper Eq. 3."""
    return 1.0 / (1.0 + s ** 2 / lam ** 2)


def compute_lambda(image, sigma=1.0, percentile=70):
    """Compute λ and the full |∇Lσ| map from the image."""
    Ls       = gaussian_filter(image, sigma=sigma)   # Lσ
    Lx       = np.gradient(Ls, axis=1)
    Ly       = np.gradient(Ls, axis=0)
    grad_mag = np.sqrt(Lx ** 2 + Ly ** 2)
    lam      = np.percentile(grad_mag, percentile)
    return lam, grad_mag


def plot_conductance(image, ch_name, out_dir):
    lam, grad_mag = compute_lambda(image)
    print(f"  λ (70th %ile of |∇Lσ|) = {lam:.6f}")
    gmax = grad_mag.max() * 1.1
    s    = np.linspace(0, gmax, 600)
    step = max(1, image.shape[0] // 512)
    DPI  = 200

    def save(fig, name):
        out = os.path.join(out_dir, name)
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        print(f"  Saved  : {out}")
        plt.close(fig)

    # ── 1a: g₂ curve ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(s, g2(s, lam), color='#534AB7', lw=2.5,
            label=r'$g_2 = 1\,/\,(1 + |\nabla L_\sigma|^2\,/\,\lambda^2)$')
    ax.axvline(lam, color='#BA7517', lw=1.5, ls='--', label=f'λ = {lam:.4f}  (70th %ile)')
    ax.axhline(0.5, color='gray', lw=0.8, ls=':')
    ax.scatter([0],   [1.0],          color='#1D9E75', zorder=5, s=80)
    ax.scatter([lam], [g2(lam, lam)], color='#534AB7', zorder=5, s=80)
    ax.annotate('g₂(0) = 1\nflat pixel → diffuse freely',
                xy=(0, 1.0), xytext=(lam * 0.6, 0.87), fontsize=10, color='#085041',
                arrowprops=dict(arrowstyle='->', color='#085041', lw=1.0))
    ax.annotate('g₂(λ) = 0.5\nthreshold point',
                xy=(lam, 0.5), xytext=(lam * 1.5, 0.58), fontsize=10, color='#3C3489',
                arrowprops=dict(arrowstyle='->', color='#534AB7', lw=1.0))
    ax.fill_between(s, 0, g2(s, lam), where=(s <= lam),
                    alpha=0.10, color='green', label='diffuse  (|∇Lσ| ≤ λ)')
    ax.fill_between(s, 0, g2(s, lam), where=(s > lam),
                    alpha=0.10, color='red',   label='block    (|∇Lσ| > λ)')
    ax.set_xlabel('|∇Lσ|  gradient of Gaussian-smoothed image', fontsize=12)
    ax.set_ylabel('conductance  c = g₂', fontsize=12)
    ax.set_title(f'AKAZE conductance function g₂  [paper Eq. 3]\nChannel: {ch_name}  |  λ = {lam:.5f}', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.15)
    ax.set_xlim(0, gmax); ax.set_ylim(0, 1.08)
    plt.tight_layout()
    save(fig, f'1a_conductance_curve_{ch_name}.png')

    # ── 1b: conductance map overlaid on image ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    cond_overlay = g2(grad_mag[::step, ::step], lam)
    ax.imshow(image[::step, ::step], cmap='gray', vmin=0, vmax=1)
    im = ax.imshow(1 - cond_overlay, cmap='hot', alpha=0.55, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='1 − g₂  (bright = diffusion blocked)')
    ax.set_title(f'Conductance map — {ch_name}\nbright = strong edge, diffusion stopped', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    save(fig, f'1b_conductance_map_{ch_name}.png')

    # ── 1c: gradient histogram with λ at 70th percentile ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    flat = grad_mag.ravel()
    cnts, edges = np.histogram(flat, bins=80)
    ctrs  = 0.5 * (edges[:-1] + edges[1:])
    cumul = np.cumsum(cnts) / cnts.sum()
    bcols = ['#534AB7' if c <= lam else '#1D9E75' for c in ctrs]
    ax.bar(ctrs, cnts, width=edges[1] - edges[0], color=bcols, alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(ctrs, cumul, color='#E24B4A', lw=1.5, ls='--', label='cumulative')
    ax2.axhline(0.70, color='#BA7517', lw=1.0, ls=':', alpha=0.8)
    ax2.set_ylabel('cumulative fraction', fontsize=11, color='#E24B4A')
    ax2.set_ylim(0, 1.05)
    ax.axvline(lam, color='#BA7517', lw=2, ls='--', label=f'λ = {lam:.4f}  (70th %ile)')
    ax.set_xlabel('|∇Lσ|  pixel gradient magnitudes', fontsize=12)
    ax.set_ylabel('pixel count', fontsize=12)
    ax.set_title(f'Gradient histogram — {ch_name}\nλ = 70th percentile  (AKAZE auto-calibration)', fontsize=12)
    ax.text(lam * 0.3, cnts.max() * 0.55, '70% below λ\n→ diffuse', fontsize=10, color='#3C3489', ha='center')
    ax.text(lam * 1.9, cnts.max() * 0.25, '30% above λ\n→ block',   fontsize=10, color='#085041', ha='center')
    l1, b1 = ax.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, b1 + b2, fontsize=10)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    save(fig, f'1c_conductance_histogram_{ch_name}.png')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DETERMINANT OF HESSIAN (DoH)
# ─────────────────────────────────────────────────────────────────────────────
#
# H = | Lxx Lxy |   det(H) = Lxx·Lyy − Lxy²
#     | Lxy Lyy |
#
# det(H) > 0  →  blob     (same-sign curvature both axes)
# det(H) < 0  →  saddle   (opposite curvature, edge-like)
# Keypoints = spatial maxima of det(H) above threshold.

def compute_hessian(image):
    Lxx  = np.gradient(np.gradient(image, axis=1), axis=1)
    Lyy  = np.gradient(np.gradient(image, axis=0), axis=0)
    Lxy  = np.gradient(np.gradient(image, axis=1), axis=0)
    detH = Lxx * Lyy - Lxy ** 2
    return Lxx, Lyy, Lxy, detH


def find_maxima(response, threshold, min_dist=5, border=24):
    """
    Non-maximum suppression with a border exclusion margin.
    border: pixels to ignore around the image edge — prevents boundary
    artifacts (padding zeros, conform_slice black regions) from being
    picked up as spurious keypoints.
    """
    from scipy.ndimage import maximum_filter
    lm   = maximum_filter(response, size=min_dist*2+1)
    mask = (response == lm) & (response > threshold)
    # zero out the border band
    mask[:border,  :]  = False
    mask[-border:, :]  = False
    mask[:,  :border]  = False
    mask[:, -border:]  = False
    r, c = np.where(mask)
    return list(zip(r, c, response[r, c]))


def plot_doh(image, ch_name, out_dir):
    step = max(1, image.shape[0] // 512)
    img  = image[::step, ::step]
    print(f"  Working image: {img.shape}  (1/{step} of original)")

    Lxx, Lyy, Lxy, detH = compute_hessian(img)
    kps = find_maxima(detH, threshold=detH.max() * 0.10, min_dist=6, border=24)
    print(f"  Keypoints found: {len(kps)}  (border-excluded)")

    DPI = 200

    def save(fig, name):
        out = os.path.join(out_dir, name)
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        print(f"  Saved  : {out}")
        plt.close(fig)

    def imsave(data, title, name, cmap='gray', cb=False):
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(data, cmap=cmap, origin='upper')
        ax.set_title(f'{title}\nChannel: {ch_name}', fontsize=12)
        ax.axis('off')
        if cb:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        save(fig, name)

    # ── 2a: raw channel image ─────────────────────────────────────────────────
    imsave(img,  f'{ch_name}  (log-stretched, uint8-normalised)',
           f'2a_image_{ch_name}.png', 'gray')

    # ── 2b-2d: second derivatives ─────────────────────────────────────────────
    imsave(Lxx, 'Lxx = ∂²L/∂x²  (horizontal curvature)',
           f'2b_Lxx_{ch_name}.png', 'RdBu_r', cb=True)
    imsave(Lyy, 'Lyy = ∂²L/∂y²  (vertical curvature)',
           f'2c_Lyy_{ch_name}.png', 'RdBu_r', cb=True)
    imsave(Lxy, 'Lxy = ∂²L/∂x∂y  (cross derivative)',
           f'2d_Lxy_{ch_name}.png', 'RdBu_r', cb=True)

    # ── 2e: det(H) heatmap ────────────────────────────────────────────────────
    imsave(detH, 'det(H) = Lxx·Lyy − Lxy²  (bright = blob response)',
           f'2e_detH_{ch_name}.png', 'hot', cb=True)

    # ── 2f: keypoints overlaid ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray', origin='upper')
    for (r, c, v) in kps:
        radius = 4 + 18 * (v / (detH.max() + 1e-9))
        ax.add_patch(plt.Circle((c, r), radius, fill=False,
                                edgecolor='#7F77DD', lw=1.2))
        ax.plot(c, r, '+', color='#CECBF6', markersize=4, markeredgewidth=1.0)
    ax.set_title(f'Keypoints detected: {len(kps)}\nCircle radius ∝ response strength  |  Channel: {ch_name}', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    save(fig, f'2f_keypoints_{ch_name}.png')

    # ── 2g: Lxx vs Lyy scatter ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    samp  = np.random.choice(img.size, size=min(5000, img.size), replace=False)
    lxx_s = Lxx.ravel()[samp]; lyy_s = Lyy.ravel()[samp]
    det_s = detH.ravel()[samp]
    ax.scatter(lxx_s, lyy_s, c=np.where(det_s > 0, '#534AB7', '#E24B4A'),
               s=4, alpha=0.4)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Lxx', fontsize=12); ax.set_ylabel('Lyy', fontsize=12)
    ax.set_title(f'Lxx vs Lyy  —  {ch_name}\npurple = blob (det>0)   red = saddle (det<0)', fontsize=12)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    save(fig, f'2g_lxx_lyy_scatter_{ch_name}.png')

    # ── 2h: cross-section through strongest keypoint ──────────────────────────
    if kps:
        r_best = max(kps, key=lambda x: x[2])[0]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax2_ = ax.twinx()
        ax.plot(img[r_best],   color='gray',    lw=1.5, label='intensity')
        ax2_.plot(detH[r_best], color='#534AB7', lw=2.5, label='det(H)')
        ax2_.axhline(detH.max() * 0.10, color='orange', lw=1.2, ls='--', label='threshold (10%)')
        ax.set_xlabel('x pixel', fontsize=12)
        ax.set_ylabel('intensity', fontsize=12, color='gray')
        ax2_.set_ylabel('det(H)', fontsize=12, color='#534AB7')
        ax.set_title(f'Cross-section at row {r_best}  |  Channel: {ch_name}', fontsize=12)
        l1, b1 = ax.get_legend_handles_labels()
        l2, b2 = ax2_.get_legend_handles_labels()
        ax.legend(l1 + l2, b1 + b2, fontsize=10)
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        save(fig, f'2h_crosssection_{ch_name}.png')

    return kps, img


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — M-LDB DESCRIPTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_patch(image, cy, cx, size=48):
    h = size // 2
    r0,r1 = max(0,cy-h), min(image.shape[0],cy+h)
    c0,c1 = max(0,cx-h), min(image.shape[1],cx+h)
    p = image[r0:r1, c0:c1]
    if p.shape != (size, size):
        pad = np.zeros((size,size), dtype=p.dtype)
        pad[:p.shape[0],:p.shape[1]] = p
        return pad
    return p.copy()


def cell_avg(patch, row, col, n=3):
    H,W = patch.shape
    ch,cw = H//n, W//n
    return patch[row*ch:(row+1)*ch, col*cw:(col+1)*cw].mean()


def mldb_bits(patch, n=3):
    avgs = {(r,c): cell_avg(patch,r,c,n) for r in range(n) for c in range(n)}
    pairs = []
    for r in range(n):
        for c in range(n):
            if c+1<n: pairs.append(((r,c),(r,c+1)))
            if r+1<n: pairs.append(((r,c),(r+1,c)))
            if r+1<n and c+1<n: pairs.append(((r,c),(r+1,c+1)))
    return [(1 if avgs[a]>avgs[b] else 0, a, b, avgs[a], avgs[b]) for a,b in pairs]


def plot_mldb(image, kps, ch_name, out_dir):
    if not kps:
        print("  No keypoints — skipping M-LDB plot.")
        return

    def patch_mean(kp):
        return extract_patch(image, kp[0], kp[1], 48).mean()

    best    = max(kps, key=patch_mean)
    r, c, _ = best
    print(f"  M-LDB patch at ({c},{r})  patch_mean={patch_mean(best):.4f}")
    patch   = extract_patch(image, r, c, 48)
    bits    = mldb_bits(patch, 3)
    bvals   = [b[0] for b in bits]
    cs      = 48 // 3
    DPI     = 200

    def save(fig, name):
        out = os.path.join(out_dir, name)
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        print(f"  Saved  : {out}")
        plt.close(fig)

    # ── 3a: raw patch ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(patch, cmap='gray', vmin=0, vmax=1)
    ax.plot(24, 24, '+', color='red', markersize=14, markeredgewidth=2.5)
    ax.set_title(f'Extracted patch around keypoint ({c},{r})\nChannel: {ch_name}', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    save(fig, f'3a_patch_{ch_name}.png')

    # ── 3b: 3×3 grid with cell averages ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(patch, cmap='gray', vmin=0, vmax=1, alpha=0.6)
    for i in range(1, 3):
        ax.axhline(i * cs, color='#7F77DD', lw=2.0)
        ax.axvline(i * cs, color='#7F77DD', lw=2.0)
    for row in range(3):
        for col in range(3):
            v = cell_avg(patch, row, col, 3)
            ax.text(col*cs+cs//2, row*cs+cs//2, f'{v:.3f}',
                    ha='center', va='center', fontsize=12, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#534AB7',
                              alpha=0.85, edgecolor='none'))
    ax.set_title(f'3×3 grid — cell average intensities\nChannel: {ch_name}  |  keypoint ({c},{r})', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    save(fig, f'3b_cell_averages_{ch_name}.png')

    # ── 3c: pair comparison arrows ────────────────────────────────────────────
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(patch, cmap='gray', vmin=0, vmax=1, alpha=0.4)
    for i in range(1, 3):
        ax.axhline(i * cs, color='#7F77DD', lw=1.0, alpha=0.5)
        ax.axvline(i * cs, color='#7F77DD', lw=1.0, alpha=0.5)
    for bit, a, b, *_ in bits:
        ax.annotate('', xy=(b[1]*cs+cs//2, b[0]*cs+cs//2),
                    xytext=(a[1]*cs+cs//2, a[0]*cs+cs//2),
                    arrowprops=dict(arrowstyle='->', lw=1.8,
                                    color='#E85D24' if bit else '#378ADD'))
    ax.legend(handles=[Line2D([0],[0],color='#E85D24',lw=2,label='bit=1  avg(A) > avg(B)'),
                        Line2D([0],[0],color='#378ADD',lw=2,label='bit=0  avg(A) ≤ avg(B)')],
              fontsize=11, loc='lower right')
    ax.set_title(f'Cell-pair comparisons → bits\nChannel: {ch_name}  |  keypoint ({c},{r})', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    save(fig, f'3c_pair_comparisons_{ch_name}.png')

    # ── 3d: binary descriptor grid ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')
    cols_per_row = 6
    for idx, bv in enumerate(bvals):
        col_ = idx % cols_per_row
        row_ = idx // cols_per_row
        fc   = '#534AB7' if bv else '#B5D4F4'
        tc   = 'white'   if bv else '#0C447C'
        ax.add_patch(plt.Rectangle([col_*0.16, 1.0 - row_*0.22 - 0.18],
                                    0.14, 0.18, facecolor=fc,
                                    edgecolor='white', lw=1.0))
        ax.text(col_*0.16+0.07, 1.0-row_*0.22-0.09, str(bv),
                ha='center', va='center', fontsize=14, color=tc, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
    ax.set_title(f'Binary descriptor  ({len(bvals)} bits)\nChannel: {ch_name}  |  keypoint ({c},{r})', fontsize=12)
    plt.tight_layout()
    save(fig, f'3d_binary_descriptor_{ch_name}.png')

    # ── 3e: multi-scale grids ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(patch, cmap='gray', vmin=0, vmax=1, alpha=0.55)
    for n_c, color, lw in [(2,'#1D9E75',2.5),(3,'#534AB7',1.8),(4,'#E85D24',1.2)]:
        sz = 48 / n_c
        for i in range(1, n_c):
            ax.axhline(i*sz, color=color, lw=lw, label=f'{n_c}×{n_c}' if i==1 else '')
            ax.axvline(i*sz, color=color, lw=lw)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_title(f'Multi-scale grids  (2×2, 3×3, 4×4)\nChannel: {ch_name}  |  keypoint ({c},{r})', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    save(fig, f'3e_multiscale_grids_{ch_name}.png')

    # ── 3f: Hamming distance matrix ───────────────────────────────────────────
    top4   = sorted(kps, key=lambda kp: extract_patch(image,kp[0],kp[1],48).mean(),
                    reverse=True)[:4]
    descs  = [mldb_bits(extract_patch(image,kr,kc,48),3) for (kr,kc,_) in top4]
    labels = [f'KP{i+1} ({kc},{kr})' for i,(kr,kc,_) in enumerate(top4)]
    nk     = len(descs)
    dm     = np.zeros((nk, nk))
    for i in range(nk):
        for j in range(nk):
            dm[i,j] = sum(a[0]!=b[0] for a,b in zip(descs[i],descs[j]))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dm, cmap='YlOrRd', vmin=0, vmax=len(bvals))
    for i in range(nk):
        for j in range(nk):
            ax.text(j, i, f'{int(dm[i,j])}', ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white' if dm[i,j] > len(bvals)*0.5 else 'black')
    ax.set_xticks(range(nk)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(nk)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(f'Hamming distance matrix  (top-4 tissue keypoints)\n0 = identical  |  max = {len(bvals)} bits', fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Hamming distance')
    plt.tight_layout()
    save(fig, f'3f_hamming_matrix_{ch_name}.png')

    # ── 3g: Heaviside bit decision ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ca = cell_avg(patch, 0, 0, 3)
    br = np.linspace(0, 1, 300)
    ax.step(br, (ca > br).astype(float), color='#534AB7', lw=3.0, where='post')
    ax.axvline(ca, color='#E24B4A', lw=2.0, ls='--', label=f'cell A avg = {ca:.3f}')
    ax.fill_between(br, 0, (ca > br).astype(float), alpha=0.12, color='#534AB7')
    ax.set_xlabel('Cell B average intensity', fontsize=12)
    ax.set_ylabel('bit  b = H(A − B)', fontsize=12)
    ax.set_title(f'Bit decision — Heaviside step function\nb = 1 if avg(A) > avg(B)  |  Channel: {ch_name}', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.15)
    ax.set_xlim(0, 1); ax.set_ylim(-0.1, 1.3); ax.set_yticks([0, 1])
    plt.tight_layout()
    save(fig, f'3g_heaviside_{ch_name}.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='AKAZE explainer — reads from TMA_Cores_Grouped_Rotate_Conformed'
    )
    p.add_argument('--data_root', type=str, required=True,
                   help='Path to TMA_Cores_Grouped_Rotate_Conformed directory')
    p.add_argument('--core_name', type=str, required=True,
                   help='Core subfolder, e.g. CORE_001')
    p.add_argument('--slice_idx', type=int, default=0,
                   help='Slice index in sorted TMA_<N> list (default 0)')
    p.add_argument('--channel', default='CK',
                   help='Channel index or name: DAPI CD31 GAP43 NFP CD3 CD163 CK AF  (default CK)')
    p.add_argument('--out_dir', type=str, default='.',
                   help='Output directory for PNGs (default: current dir)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("AKAZE Core Functions — Visual Explainer")
    print("=" * 60)
    print(f"  data_root : {args.data_root}")
    print(f"  core_name : {args.core_name}")
    print(f"  slice_idx : {args.slice_idx}")
    print(f"  channel   : {args.channel}")
    print(f"  out_dir   : {args.out_dir}")
    print()

    arr, fpath = load_slice(args.data_root, args.core_name, args.slice_idx)
    image, ch_name = extract_channel(arr, args.channel)
    print(f"  Image    : {image.shape}  range [{image.min():.3f}, {image.max():.3f}]")
    print()

    print("[1/3] Conductance function g₂  [paper Eq. 3]")
    plot_conductance(image, ch_name, args.out_dir)
    print()

    print("[2/3] Determinant of Hessian blob detector")
    kps, img_ds = plot_doh(image, ch_name, args.out_dir)
    print()

    print("[3/3] M-LDB descriptor")
    plot_mldb(img_ds, kps, ch_name, args.out_dir)
    print()

    print("Done. PNGs saved to:", args.out_dir)