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
    cv2 Triangle: peak = histogram max. Tail = whichever extreme (bin 0
    or bin N-1) is farther from peak. Chord runs peak->tail (tail count
    forced to 0). Threshold = bin between peak and tail with maximum
    perpendicular distance from that chord.

    Returns
    -------
    bin_centres   : bin-centre values (length N)
    peak_idx      : index of peak bin
    tail_idx      : index of tail bin (0 or N-1)
    threshold     : bin-centre of threshold
    threshold_idx : integer index of threshold bin
    distances     : perpendicular distances (non-zero between peak & tail)
    """
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n = len(histogram)
    peak_idx = int(np.argmax(histogram))
    if peak_idx <= (n - 1 - peak_idx):
        tail_idx = n - 1
        b_lo, b_hi = peak_idx, tail_idx
    else:
        tail_idx = 0
        b_lo, b_hi = tail_idx, peak_idx
    px, py = bin_centres[peak_idx], float(histogram[peak_idx])
    tx, ty = bin_centres[tail_idx], 0.0
    cdx = tx - px
    cdy = ty - py
    length = math.sqrt(cdx * cdx + cdy * cdy) + 1e-12
    distances = np.zeros(n, dtype=float)
    for b in range(b_lo, b_hi + 1):
        xb = bin_centres[b]
        yb = float(histogram[b])
        distances[b] = abs(cdy * (xb - px) - cdx * (yb - py)) / length
    threshold_idx = int(np.argmax(distances[b_lo:b_hi + 1])) + b_lo
    return (bin_centres, peak_idx, tail_idx,
            bin_centres[threshold_idx], threshold_idx, distances)


# ─────────────────────────────────────────────────────────────────────────────
# main figure function
# ─────────────────────────────────────────────────────────────────────────────

def build_segmentation_figure(file_path: str, out_path: str, slide_index: int = 0):
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

        n_levels     = len(series0.levels)
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

    hist_counts, hist_edges = np.histogram(blur.ravel(), bins=256, range=(0, 255))
    (bin_centres, peak_idx, tail_idx,
     tri_threshold, threshold_idx, distances) = _triangle_geometry(hist_counts.astype(float), hist_edges)

    # ── stage (d): morphology ─────────────────────────────────────────────────
    binary_masked = cv2.bitwise_and(binary_raw, binary_raw, mask=safe_mask)
    closed        = cv2.morphologyEx(binary_masked, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=2)
    final_mask    = cv2.morphologyEx(closed,        cv2.MORPH_OPEN,  OPEN_KERNEL,  iterations=2)

    # ── stage (e): filtered candidates overlaid ───────────────────────────────
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay     = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    
    # Adaptive Processing for Slice 3
    SPLIT_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    processed_contours = []

    if (slide_index + 1) == 3:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20000:
                # Isolate the anomalous massive blob
                blob_mask = np.zeros_like(norm)
                cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
                
                # Sever artificial connections via aggressive erosion
                eroded_blob = cv2.erode(blob_mask, SPLIT_KERNEL, iterations=2)
                sub_contours, _ = cv2.findContours(eroded_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for sub_cnt in sub_contours:
                    sub_area = cv2.contourArea(sub_cnt)
                    if sub_area < 500: 
                        continue
                    
                    # Restore original core dimensions via dilation
                    piece_mask = np.zeros_like(norm)
                    cv2.drawContours(piece_mask, [sub_cnt], -1, 255, -1)
                    restored_piece = cv2.dilate(piece_mask, SPLIT_KERNEL, iterations=2)
                    restored_cnts, _ = cv2.findContours(restored_piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if restored_cnts:
                        processed_contours.append(restored_cnts[0])
            else:
                # Standard core contour
                processed_contours.append(cnt)
    else:
        # Pass-through for non-Slice 3 configurations
        processed_contours = contours

    kept, discarded_area, discarded_aspect = [], [], []

    for cnt in processed_contours:
        area   = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h if h > 0 else 0
        
        if area < params["MIN_AREA"]:
            discarded_area.append(cnt)
        elif aspect < 0.3 or aspect > 3.0:
            discarded_aspect.append(cnt)
        else:
            kept.append(cnt)

    cv2.drawContours(overlay, discarded_area,   -1, (220, 80,  80),  4) 
    cv2.drawContours(overlay, discarded_aspect, -1, (230, 160, 20),  4) 
    cv2.drawContours(overlay, kept,             -1, (60,  200, 100), 8) 

    for cnt in kept:
        M_cnt = cv2.moments(cnt)
        if M_cnt["m00"] != 0:
            cx = int(M_cnt["m10"] / M_cnt["m00"])
            cy = int(M_cnt["m01"] / M_cnt["m00"])
            cv2.circle(overlay, (cx, cy), 6, (60, 200, 100), -1)

    tma_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

   # ─────────────────────────────────────────────────────────────────────────
    # Figure layout — Split into Part 1 (Masks) and Part 2 (Results)
    # ─────────────────────────────────────────────────────────────────────────

    # Extract base path to save part1 and part2
    base_out, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".png"
    out_path_part1 = f"{base_out}_part1{ext}"
    out_path_part2 = f"{base_out}_part2{ext}"

    MASK_CMAP  = "inferno"
    GRAY_CMAP  = "gray"
    
    # Aggressively scaled typography for A4 legibility
    LABEL_KW   = dict(fontsize=18, color="black", fontfamily="monospace", labelpad=12)
    TITLE_KW   = dict(fontsize=20, color="black", fontfamily="monospace")
    SUPTITLE_FONT = 24

    def _format_ax(ax):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return ax

    slice_num = tma_name.split('_')[-1]
    title_str = f"Core segmentation pipeline  —  Slice {slice_num}"

    # =========================================================================
    # FIGURE 1: Mask Progression (2x2)
    # =========================================================================
    fig1 = plt.figure(figsize=(14, 14))
    fig1.patch.set_facecolor("white")

    gs1 = GridSpec(
        2, 2, figure=fig1,
        hspace=0.25, wspace=0.15,
        left=0.05, right=0.95, top=0.86, bottom=0.06  # Lowered top boundary to 0.86
    )

    # ── (a) summed projection
    ax_a = _format_ax(fig1.add_subplot(gs1[0, 0]))
    ax_a.imshow(norm, cmap=GRAY_CMAP, interpolation="nearest")
    ax_a.set_title("(a)  Summed projection", **TITLE_KW)
    ax_a.set_xlabel("$I_{\\mathrm{norm}}$", **LABEL_KW)

    # ── (b) safe mask
    ax_b = _format_ax(fig1.add_subplot(gs1[0, 1]))
    ax_b.imshow(safe_mask, cmap=MASK_CMAP, interpolation="nearest")
    ax_b.set_title("(b)  Safe mask  $M_{\\mathrm{safe}}$", **TITLE_KW)
    ax_b.set_xlabel("$M_{\\mathrm{safe}}$", **LABEL_KW)

    # ── (c) triangle mask
    ax_c = _format_ax(fig1.add_subplot(gs1[1, 0]))
    ax_c.imshow(binary_raw, cmap=MASK_CMAP, interpolation="nearest")
    ax_c.set_title(f"(c)  Triangle mask  $M_{{\\mathrm{{tri}}}}$", **TITLE_KW)
    ax_c.set_xlabel("$M_{\\mathrm{tri}}$", **LABEL_KW)

    # ── (d) final mask
    ax_d = _format_ax(fig1.add_subplot(gs1[1, 1]))
    ax_d.imshow(final_mask, cmap=MASK_CMAP, interpolation="nearest")
    ax_d.set_title("(d)  Final mask  $M_{\\mathrm{final}}$", **TITLE_KW)
    ax_d.set_xlabel("$M_{\\mathrm{final}}$", **LABEL_KW)

    # Anchored suptitle higher using y=0.96
    fig1.suptitle(title_str + "\n(Part 1: Masking)", 
                  fontsize=SUPTITLE_FONT, color="black", fontfamily="monospace", weight="bold", y=0.96)
    
    plt.figure(fig1.number)
    plt.savefig(out_path_part1, dpi=200, bbox_inches="tight", facecolor=fig1.get_facecolor())
    plt.close(fig1)
    print(f"  Figure 1 saved → {out_path_part1}")

    # =========================================================================
    # FIGURE 2: Detection and Threshold Geometry (1x2)
    # =========================================================================
    fig2 = plt.figure(figsize=(16, 8))
    fig2.patch.set_facecolor("white")

    gs2 = GridSpec(
        1, 2, figure=fig2,
        wspace=0.20,
        left=0.05, right=0.95, top=0.82, bottom=0.10  # Adjusted top boundary
    )

    # ── (a) detected cores (Formerly e)
    ax_e = _format_ax(fig2.add_subplot(gs2[0, 0]))
    ax_e.imshow(overlay, interpolation="nearest")
    ax_e.set_title("(a)  Detected cores", **TITLE_KW)

    ax_e.set_xlabel(f"accepted: {len(kept)}\n"
                    f"rejected (size): {len(discarded_area)}   "
                    f"rejected (shape): {len(discarded_aspect)}",
                    fontsize=16, color="black", fontfamily="monospace") 

    legend_elements = [
        mpatches.Patch(facecolor=(60/255,  200/255, 100/255), label="accepted"),
        mpatches.Patch(facecolor=(220/255,  80/255,  80/255), label="rejected — area"),
        mpatches.Patch(facecolor=(230/255, 160/255,  20/255), label="rejected — aspect"),
    ]
    ax_e.legend(handles=legend_elements, loc="lower right",
                fontsize=15, framealpha=0.85,
                facecolor="white", edgecolor="#cccccc", labelcolor="black")

    # ── (b) histogram inset (Formerly f)
    ax_f = fig2.add_subplot(gs2[0, 1])
    ax_f.set_facecolor("white")
    for spine in ax_f.spines.values():
        spine.set_edgecolor("#cccccc")
    ax_f.tick_params(colors="black", labelsize=15)

    peak_count   = float(hist_counts[peak_idx])
    nonzero_bins = np.where(hist_counts > peak_count * 0.001)[0]
    last_bin     = int(nonzero_bins[-1]) if len(nonzero_bins) else len(bin_centres) - 1
    zoom_xmax    = min(255, max(int(bin_centres[threshold_idx]) + 20, int(bin_centres[last_bin]) + 5))
    
    ax_f.set_xlim(0, 255)
    ax_f.set_ylim(0, peak_count * 1.08)

    ax_f.bar(bin_centres, hist_counts, width=(bin_centres[1] - bin_centres[0]),
             color="#3d6bb5", alpha=0.65, label="histogram $H(b)$", zorder=2)

    px, py = float(bin_centres[peak_idx]), float(hist_counts[peak_idx])
    tx, ty = float(bin_centres[tail_idx]), 0.0                          
    cdx, cdy = tx - px, ty - py

    chord_line, = ax_f.plot([px, tx], [py, ty], color="#e69138", linewidth=2.5, linestyle="--",
                            label=r"chord  $(b_{\mathrm{peak}},H_{\mathrm{peak}})\!\to\!(b_{\mathrm{tail}},0)$",
                            zorder=4, clip_on=True)

    xb_data = float(bin_centres[threshold_idx])
    yb_data = float(hist_counts[threshold_idx])

    def _draw_perp(event, _ax=ax_f, _px=px, _py=py, _cdx=cdx, _cdy=cdy, _xb=xb_data, _yb=yb_data):
        if getattr(_ax, "_perp_drawn", False):
            return
        trans = _ax.transData
        P = trans.transform([_px, _py])
        D = trans.transform([_px + _cdx, _py + _cdy])
        B = trans.transform([_xb, _yb])
        cv = D - P
        t = np.dot(B - P, cv) / (np.dot(cv, cv) + 1e-24)
        F = P + t * cv
        xf, yf = _ax.transData.inverted().transform(F)
        _ax.annotate("", xy=(xf, yf), xytext=(_xb, _yb),
                     arrowprops=dict(arrowstyle="<->", color="#e05c5c", lw=2.5), zorder=5)
        _ax.text((_xb + xf) / 2 + 0.5, (_yb + yf) / 2,
                 r"$d(b)_{\max}$", fontsize=16, color="#e05c5c", va="center")
        _ax._perp_drawn = True
        _ax.get_figure().canvas.draw()

    fig2.canvas.mpl_connect("draw_event", _draw_perp)

    ax_f.axvline(tri_threshold, color="#e05c5c", linewidth=2.5, linestyle="-",
                 label=rf"$T_{{\mathrm{{tri}}}}$ = {tri_threshold:.0f}", zorder=4)
    
    ax_f.set_xlabel("Intensity bin  $b$", fontsize=18, color="black")
    ax_f.set_ylabel("Count  $H(b)$", fontsize=18, color="black")
    ax_f.set_title("(b)  Triangle threshold geometry", fontsize=20, color="black", loc="center")
    
    ax_f.legend(fontsize=15, framealpha=0.85, facecolor="white", edgecolor="#cccccc",
                labelcolor="black", loc="upper right")

    fig2.suptitle(title_str + "\n(Part 2: Outcomes)", 
                  fontsize=SUPTITLE_FONT, color="black", fontfamily="monospace", weight="bold", y=0.98)
    
    plt.figure(fig2.number)
    fig2.canvas.draw()  
    plt.savefig(out_path_part2, dpi=200, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  Figure 2 saved → {out_path_part2}")


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

    build_segmentation_figure(file_path, out_path, slide_index=idx)


if __name__ == "__main__":
    main()