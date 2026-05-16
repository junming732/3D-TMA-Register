"""
TMA Core Segmentation — Mask Progression Figure Generator
==========================================================

Produces a publication-quality figure panel showing the sequential
intermediate masks from the core segmentation pipeline, for use as a
methods figure.

Panel layout (one column per stage):
  (a) Summed channel projection (normalised)
  (b) Safe mask  M_safe  — coarse Otsu + dilation
  (c) Triangle mask  M_tri  — contrast-stretched + Triangle threshold
  (d) Final mask  M_final — after morphological closing/opening
  (e) Detected cores — M_final with surviving core contours overlaid

An optional inset histogram panel illustrates the Triangle threshold
geometry (chord + perpendicular distance) for the methods text.

Usage
-----
    python generate_mask_figure.py                     # uses first TMA file
    python generate_mask_figure.py --slide_index 2    # third TMA file (0-based)
    python generate_mask_figure.py --out_dir ./figs

Requirements: same environment as cropping_cores_rotate.py
    pip install matplotlib tifffile numpy opencv-python
"""

import os
import sys
import argparse
import math

import cv2
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")                   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── make sure config is importable the same way the main script does ──────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config


# ─────────────────────────────────────────────────────────────────────────────
# helpers (mirrors cropping_cores_rotate.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def apply_linear_stretch(image, low_p=0.5, high_p=99.5):
    p_min, p_max = np.percentile(image[image > 0], (low_p, high_p))
    return np.clip((image - p_min) / (p_max - p_min + 1e-5) * 255, 0, 255).astype(np.uint8)


def _triangle_geometry(histogram, bin_edges):
    """
    Returns arrays needed to draw the Triangle-threshold chord diagram.

    histogram : 1-D array of counts (length N)
    bin_edges : 1-D array of bin edges (length N+1)

    Returns
    -------
    bin_centres : centres of each bin
    peak_idx    : index of the histogram peak
    tail_idx    : index of the low-intensity tail extreme used by cv2
    threshold   : the selected threshold bin centre
    distances   : perpendicular distances from chord at each bin
    """
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Mirror cv2's Triangle: peak is the max, tail is bin 0 (low side)
    peak_idx = int(np.argmax(histogram))
    tail_idx = 0

    # Chord from peak to tail
    x1, y1 = bin_centres[tail_idx],  histogram[tail_idx]
    x2, y2 = bin_centres[peak_idx],  histogram[peak_idx]
    dx, dy  = x2 - x1, y2 - y1
    length  = math.sqrt(dx * dx + dy * dy) + 1e-12

    # Perpendicular distances for bins between tail and peak
    distances = np.zeros_like(histogram, dtype=float)
    for b in range(tail_idx, peak_idx + 1):
        xb, yb = bin_centres[b], histogram[b]
        dist = abs(dy * xb - dx * yb + x2 * y1 - y2 * x1) / length
        distances[b] = dist

    threshold_idx = int(np.argmax(distances[tail_idx:peak_idx + 1])) + tail_idx
    return bin_centres, peak_idx, tail_idx, bin_centres[threshold_idx], distances


# ─────────────────────────────────────────────────────────────────────────────
# main figure function
# ─────────────────────────────────────────────────────────────────────────────

def build_segmentation_figure(file_path: str, out_path: str):
    """
    Run the segmentation pipeline on *file_path* (single TMA slide),
    capture every intermediate mask, and save a figure to *out_path*.
    """

    params = {
        "OPEN_SIZE": 15,
        "MIN_AREA":  2000,
        "TILT_LIMIT": 12.0,
    }
    OPEN_KERNEL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["OPEN_SIZE"], params["OPEN_SIZE"]))
    CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

    # ── load ──────────────────────────────────────────────────────────────────
    with tifffile.TiffFile(file_path) as tif:
        series0  = tif.series[0]
        low_res  = series0.levels[-1]
        if low_res.shape[-1] < 100:
            low_res = series0.levels[-2]

        # pyramid metadata for caption
        n_levels     = len(series0.levels)
        # Typical TMA OME-TIFF pyramid levels: 1×, 8×, 16×, 32× downsampled
        pyramid_factors = [1, 2, 4, 8, 16, 32]
        used_level_idx  = n_levels - 1 if low_res.shape[-1] >= 100 else n_levels - 2
        used_factor     = pyramid_factors[used_level_idx] if used_level_idx < len(pyramid_factors) else "?"

        h_high = series0.levels[0].shape[-2]
        h_low  = low_res.shape[-2]
        actual_factor = round(h_high / h_low)

        raw_stack = low_res.asarray()

    combined = np.sum(raw_stack, axis=0, dtype=np.float32)

    p99 = np.percentile(combined, 99)
    if p99 < 1: p99 = combined.max()
    norm = np.clip((combined / p99) * 255.0, 0, 255).astype(np.uint8)

    # ── stage (b): safe mask ──────────────────────────────────────────────────
    kernel_bg     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
    bg_est        = cv2.morphologyEx(norm, cv2.MORPH_OPEN, kernel_bg)
    foreground_rough = cv2.subtract(norm, bg_est)
    _, rough_mask = cv2.threshold(foreground_rough, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    safe_mask     = cv2.morphologyEx(rough_mask, cv2.MORPH_DILATE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

    # ── stage (c): triangle mask ───────────────────────────────────────────────
    stretched_img = apply_linear_stretch(norm)
    blur          = cv2.GaussianBlur(stretched_img, (15, 15), 0)
    thresh_val, binary_raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    # histogram geometry for inset
    hist_counts, hist_edges = np.histogram(blur.ravel(), bins=256, range=(0, 255))
    (bin_centres, peak_idx, tail_idx,
     tri_threshold, distances) = _triangle_geometry(hist_counts.astype(float), hist_edges)

    # ── stage (d): morphology ─────────────────────────────────────────────────
    binary_masked = cv2.bitwise_and(binary_raw, binary_raw, mask=safe_mask)
    closed        = cv2.morphologyEx(binary_masked, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=2)
    final_mask    = cv2.morphologyEx(closed,        cv2.MORPH_OPEN,  OPEN_KERNEL,  iterations=2)

    # ── stage (e): filtered candidates overlaid ───────────────────────────────
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay     = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    kept, discarded_area, discarded_aspect = [], [], []

    for cnt in contours:
        area   = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h if h > 0 else 0
        if area < params["MIN_AREA"]:
            discarded_area.append(cnt)
        elif aspect < 0.3 or aspect > 3.0:
            discarded_aspect.append(cnt)
        else:
            kept.append(cnt)

    cv2.drawContours(overlay, discarded_area,   -1, (220, 80,  80),  4)   # red  — too small
    cv2.drawContours(overlay, discarded_aspect, -1, (230, 160, 20),  4)   # amber — bad aspect
    cv2.drawContours(overlay, kept,             -1, (60,  200, 100), 8)   # green — accepted

    # draw centroid dots for accepted cores
    for cnt in kept:
        M_cnt = cv2.moments(cnt)
        if M_cnt["m00"] != 0:
            cx = int(M_cnt["m10"] / M_cnt["m00"])
            cy = int(M_cnt["m01"] / M_cnt["m00"])
            # You may also optionally increase the centroid radius (e.g., from 4 to 6) 
            # so it remains proportional to the thicker contours.
            cv2.circle(overlay, (cx, cy), 6, (60, 200, 100), -1)

    tma_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

    # ─────────────────────────────────────────────────────────────────────────
    # Figure layout
    #
    #   Row 0: [a] projection    [b] safe mask    [c] triangle mask
    #   Row 1: [d] final mask    [e] detections   [f] histogram inset
    # ─────────────────────────────────────────────────────────────────────────

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white") # Shifted to white

    gs = GridSpec(
        2, 3, 
        figure=fig,
        height_ratios=[1, 1],
        hspace=0.25,          
        wspace=0.10,
        left=0.03, right=0.97,
        top=0.90,  bottom=0.06,
    )

    MASK_CMAP  = "inferno"
    GRAY_CMAP  = "gray"
    # Darkened text variables for white background contrast
    LABEL_KW   = dict(fontsize=9, color="#222222", fontfamily="monospace", labelpad=5)
    TITLE_KW   = dict(fontsize=10, color="#444444", fontfamily="monospace", style="italic")

    def _ax(row, col):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc") # Lightened spines
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return ax

    # ── (a) summed projection ─────────────────────────────────────────────────
    ax_a = _ax(0, 0)
    ax_a.imshow(norm, cmap=GRAY_CMAP, interpolation="nearest")
    ax_a.set_title(f"(a)  Summed projection\n"
                   f"({actual_factor}× downsampled, level {used_level_idx + 1}/{n_levels})",
                   **TITLE_KW)
    ax_a.set_xlabel("$I_{\\mathrm{norm}}$", **LABEL_KW)

    # ── (b) safe mask ─────────────────────────────────────────────────────────
    ax_b = _ax(0, 1)
    ax_b.imshow(safe_mask, cmap=MASK_CMAP, interpolation="nearest")
    ax_b.set_title("(b)  Safe mask  $M_{\\mathrm{safe}}$\n"
                   "Otsu on BG-subtracted proj. + dilation",
                   **TITLE_KW)
    ax_b.set_xlabel("$M_{\\mathrm{safe}}$", **LABEL_KW)

    # ── (c) triangle mask ─────────────────────────────────────────────────────
    ax_c = _ax(0, 2)
    ax_c.imshow(binary_raw, cmap=MASK_CMAP, interpolation="nearest")
    ax_c.set_title(f"(c)  Triangle mask  $M_{{\\mathrm{{tri}}}}$\n"
                   f"$T_{{\\mathrm{{tri}}}}$ = {thresh_val:.0f}  (contrast-stretched + Gaussian blur)",
                   **TITLE_KW)
    ax_c.set_xlabel("$M_{\\mathrm{tri}}$", **LABEL_KW)

    # ── (d) final mask ────────────────────────────────────────────────────────
    ax_d = _ax(1, 0)
    ax_d.imshow(final_mask, cmap=MASK_CMAP, interpolation="nearest")
    ax_d.set_title("(d)  Final mask  $M_{\\mathrm{final}}$\n"
                   "$M_{\\mathrm{safe}}$ $\\cap$ $M_{\\mathrm{tri}}$  →  close  →  open",
                   **TITLE_KW)
    ax_d.set_xlabel("$M_{\\mathrm{final}}$", **LABEL_KW)

    # ── (e) detected cores ────────────────────────────────────────────────────
    ax_e = _ax(1, 1)
    ax_e.imshow(overlay, interpolation="nearest")
    ax_e.set_title(f"(e)  Detected cores\n"
                   f"area $\\geq$ {params['MIN_AREA']} px,  "
                   f"aspect ratio $\\in$ [0.3, 3.0]",
                   **TITLE_KW)
    # [Keep ax_e.imshow and ax_e.set_title code...]

    ax_e.set_xlabel(f"accepted: {len(kept)}   "
                    f"rejected (size): {len(discarded_area)}   "
                    f"rejected (shape): {len(discarded_aspect)}",
                    fontsize=10, color="#444444", fontfamily="monospace") # Darkened

    legend_elements = [
        mpatches.Patch(facecolor=(60/255,  200/255, 100/255), label="accepted"),
        mpatches.Patch(facecolor=(220/255,  80/255,  80/255), label="rejected — area"),
        mpatches.Patch(facecolor=(230/255, 160/255,  20/255), label="rejected — aspect"),
    ]
    ax_e.legend(handles=legend_elements, loc="lower right",
                fontsize=10, framealpha=0.85,
                facecolor="white", edgecolor="#cccccc", # Updated for light theme
                labelcolor="#222222")

    # ── (f) histogram inset — Triangle threshold geometry ─────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.set_facecolor("white")
    for spine in ax_f.spines.values():
        spine.set_edgecolor("#cccccc")
    ax_f.tick_params(colors="#444444", labelsize=7.5)

    # histogram bars
    ax_f.bar(bin_centres, hist_counts, width=(bin_centres[1] - bin_centres[0]),
             color="#3d6bb5", alpha=0.65, label="histogram $H(b)$", zorder=2)

    # chord from tail to peak (Color shifted to high-contrast orange)
    ax_f.plot(
        [bin_centres[tail_idx], bin_centres[peak_idx]],
        [hist_counts[tail_idx], hist_counts[peak_idx]],
        color="#e69138", linewidth=1.4, linestyle="--",
        label="chord  $(b_{\\mathrm{min}}, H(b_{\\mathrm{min}}))$ → $(b_{\\mathrm{max}}, H(b_{\\mathrm{max}}))$",
        zorder=4,
    )

    # perpendicular distance line at threshold
    t_idx = int(np.argmax(distances))
    x1c, y1c = bin_centres[tail_idx],  hist_counts[tail_idx]
    x2c, y2c = bin_centres[peak_idx],  hist_counts[peak_idx]
    xb, yb = bin_centres[t_idx], hist_counts[t_idx]
    dx, dy  = x2c - x1c, y2c - y1c
    t_param = ((xb - x1c) * dx + (yb - y1c) * dy) / (dx * dx + dy * dy + 1e-12)
    xf = x1c + t_param * dx
    yf = y1c + t_param * dy
    ax_f.annotate(
        "", xy=(xf, yf), xytext=(xb, yb),
        arrowprops=dict(arrowstyle="<->", color="#e05c5c", lw=1.4),
        zorder=5,
    )
    ax_f.text(xb + 1, (yb + yf) / 2, "$d(b)_{\\max}$",
              fontsize=10, color="#e05c5c", va="center")

    # threshold vertical line
    ax_f.axvline(tri_threshold, color="#e05c5c", linewidth=1.6, linestyle="-",
                 label=f"$T_{{\\mathrm{{tri}}}}$ = {tri_threshold:.0f}", zorder=4)

    ax_f.set_xlabel("Intensity bin  $b$",  fontsize=8, color="#444444")
    ax_f.set_ylabel("Count  $H(b)$",       fontsize=8, color="#444444")
    ax_f.set_title(
        "(f)  Triangle threshold geometry  ($T_{\\mathrm{tri}} = \\arg\\max_b\\, d(b)$)",
        fontsize=8, color="#444444", loc="left",
    )
    ax_f.legend(fontsize=10, framealpha=0.85,
                facecolor="white", edgecolor="#cccccc",
                labelcolor="#222222", loc="upper right")
    
    # HORIZONTAL ZOOM LOGIC
    # Stretch out the X-axis to make the geometry visible. Focus on the region slightly past the peak.
    zoom_xmax = min(255, max(50, peak_idx * 3))
    ax_f.set_xlim(0, zoom_xmax)

    # ── global title ──────────────────────────────────────────────────────────
    # Extract the slice number from the end of the directory string
    slice_num = tma_name.split('_')[-1]
    
    fig.suptitle(
        f"Core segmentation pipeline  —  Slice {slice_num}",
        fontsize=14, color="black", fontfamily="monospace", y=0.975,
    )

    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate TMA mask-progression figure for methods section."
    )
    parser.add_argument(
        "--slide_index", type=int, default=0,
        help="Index into config.TMA_FILES to use (default: 0 = first slide).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=".",
        help="Directory to write the PNG figure (default: current directory).",
    )
    parser.add_argument(
        "--filename", type=str, default="fig_segmentation_pipeline.png",
        help="Output filename (default: fig_segmentation_pipeline.png).",
    )
    args = parser.parse_args()

    if not hasattr(config, "TMA_FILES") or len(config.TMA_FILES) == 0:
        sys.exit("config.TMA_FILES is empty — nothing to process.")

    idx = args.slide_index
    if idx >= len(config.TMA_FILES):
        sys.exit(f"slide_index {idx} out of range "
                 f"(config.TMA_FILES has {len(config.TMA_FILES)} entries).")

    file_path = config.TMA_FILES[idx]
    if not os.path.exists(file_path):
        sys.exit(f"File not found: {file_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.filename)

    tma_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    print(f"Building segmentation figure for: {tma_name}")

    build_segmentation_figure(file_path, out_path)


if __name__ == "__main__":
    main()