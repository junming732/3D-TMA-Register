"""
held_out_tre.py — held-out-landmark Target Registration Error.
======================================================================
WHY THIS EXISTS: VALIS's own automatic fallback (used when manual
ground-truth landmarks aren't available) measures error using the same
matched keypoints that were used to FIT the transform — that's closer
to training error than validation error, since the transform was
literally optimized to minimize distance on those exact points, and
will read more optimistic than true generalization accuracy.

This module does the stricter version instead: correspondences are
split into a FIT set and a HELD-OUT set BEFORE any transform is fitted.
The transform never sees the held-out points. TRE is then computed only
on the held-out set, which is a real (if still auto-detected, not
manually-annotated) validation error rather than a residual-on-training-
points number.

CRITICAL FOR FAIR COMPARISON: when comparing two candidates (e.g.
RANSAC vs MAGSAC), use the SAME held-out split for both — split once,
fit each candidate on the same fit set, evaluate both on the identical
held-out points. Otherwise you're not isolating the variable you think
you're isolating (see integration into test_ransac_vs_magsac.py for the
pattern).

Two evaluation modes:
  - compute_tre_affine(M, ...)   — for an affine transform (L0)
  - compute_tre_warp(map_x, map_y, ...) — for a dense deformation field (L1),
    bilinearly sampling the field at each held-out point

No GPU/torch dependency — pure numpy/opencv.
"""

import numpy as np
import cv2


def split_correspondences_for_tre(n_matches, holdout_frac=0.2, min_holdout=15,
                                   min_fit=None, seed=None):
    """
    Partition n_matches correspondence indices into disjoint (fit_idx,
    holdout_idx). Held-out points are NEVER used for fitting — only for
    TRE evaluation afterward.

    min_fit: minimum points required in the fit set (defaults to your
    pipeline's MIN_MATCHES if not specified — pass it in explicitly to
    keep this in sync with your production config rather than
    duplicating the constant here).

    Returns (fit_idx, holdout_idx), or (None, None) if there aren't
    enough total matches to support both a valid fit set and a
    min_holdout-sized held-out set.
    """
    if min_fit is None:
        min_fit = 20  # mirrors MIN_MATCHES default; pass explicitly to override
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_matches)

    n_holdout = max(min_holdout, int(round(n_matches * holdout_frac)))
    n_fit = n_matches - n_holdout
    if n_fit < min_fit:
        # Not enough points to do both a reliable fit AND a held-out set.
        # Shrink the held-out set rather than silently degrading the fit.
        n_holdout = n_matches - min_fit
        if n_holdout < min_holdout:
            return None, None  # genuinely too few matches for this pair

    holdout_idx = idx[:n_holdout]
    fit_idx = idx[n_holdout:]
    return fit_idx, holdout_idx


def compute_tre_affine(M, src_holdout, dst_holdout, px_size_um=None, image_diag_px=None):
    """
    M: affine 2x3, in the same convention used throughout this pipeline —
    fit via cv2.estimateAffine2D(dst_pts, src_pts, ...), i.e. M maps
    moving(dst)-space points into fixed(src)-space: pred_src = M @ [dst, 1].
    src_holdout, dst_holdout: held-out correspondence points, (N, 2) or
    (N, 1, 2) — NOT used in fitting M.

    Returns a dict: per-point distances (px, and um if px_size_um given),
    rTRE (if image_diag_px given), and summary stats matching the
    "Global TRE summary" format already used in the report.
    """
    dst_flat = np.asarray(dst_holdout).reshape(-1, 2).astype(np.float64)
    src_flat = np.asarray(src_holdout).reshape(-1, 2).astype(np.float64)

    pred_src = (M[:, :2] @ dst_flat.T).T + M[:, 2]
    dists_px = np.linalg.norm(pred_src - src_flat, axis=1)

    return _summarize_tre(dists_px, px_size_um, image_diag_px)


def compute_tre_warp(map_x, map_y, src_holdout, dst_holdout, px_size_um=None, image_diag_px=None):
    """
    Same idea as compute_tre_affine, but for a dense deformation field
    (map_x, map_y — same convention as elsewhere in this pipeline:
    map_x[y, x]/map_y[y, x] give the moving-image coordinate that fixed-
    image pixel (x, y) corresponds to). Held-out FIXED points are looked
    up in the field via bilinear interpolation to get predicted moving
    coordinates, then compared against the actual held-out MOVING
    correspondence.

    src_holdout: held-out points in FIXED-image space, (N, 2) — used to
        index into the field.
    dst_holdout: the corresponding held-out points in MOVING-image
        space — ground truth to compare the field's prediction against.
    """
    src_flat = np.asarray(src_holdout).reshape(-1, 2).astype(np.float64)
    dst_flat = np.asarray(dst_holdout).reshape(-1, 2).astype(np.float64)

    h, w = map_x.shape
    # Bilinear sample map_x/map_y at each held-out fixed-space point.
    xs = np.clip(src_flat[:, 0], 0, w - 1.001)
    ys = np.clip(src_flat[:, 1], 0, h - 1.001)
    x0 = np.floor(xs).astype(int); x1 = x0 + 1
    y0 = np.floor(ys).astype(int); y1 = y0 + 1
    wx = xs - x0; wy = ys - y0

    def bilerp(field):
        v00 = field[y0, x0]; v01 = field[y0, x1]
        v10 = field[y1, x0]; v11 = field[y1, x1]
        return (v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy)
              + v10 * (1 - wx) * wy       + v11 * wx * wy)

    pred_x = bilerp(map_x)
    pred_y = bilerp(map_y)
    pred_dst = np.stack([pred_x, pred_y], axis=1)

    dists_px = np.linalg.norm(pred_dst - dst_flat, axis=1)
    return _summarize_tre(dists_px, px_size_um, image_diag_px)


def _summarize_tre(dists_px, px_size_um=None, image_diag_px=None):
    dists_px = np.asarray(dists_px, dtype=np.float64)
    dists_px = dists_px[~np.isnan(dists_px)]
    result = dict(
        n=len(dists_px),
        mean_px=dists_px.mean() if len(dists_px) else np.nan,
        median_px=np.median(dists_px) if len(dists_px) else np.nan,
        q3_px=np.percentile(dists_px, 75) if len(dists_px) else np.nan,
        p90_px=np.percentile(dists_px, 90) if len(dists_px) else np.nan,
        max_px=dists_px.max() if len(dists_px) else np.nan,
        std_px=dists_px.std() if len(dists_px) else np.nan,
        per_point_px=dists_px,
    )
    if px_size_um is not None:
        for k in ["mean", "median", "q3", "p90", "max", "std"]:
            result[f"{k}_um"] = result[f"{k}_px"] * px_size_um
    if image_diag_px is not None:
        result["mean_rtre"] = result["mean_px"] / image_diag_px
        result["median_rtre"] = result["median_px"] / image_diag_px
        result["per_point_rtre"] = dists_px / image_diag_px
    return result


def detect_independent_sift_landmarks(fixed_img, moving_img, fixed_mask=None, moving_mask=None,
                                       lowe_ratio=0.8, ransac_thresh=15.0, min_matches=15):
    """
    SIFT detection/matching, entirely independent of whatever detector the
    production pipeline uses for fitting (AKAZE + BFMatcher-Hamming
    elsewhere in this codebase). A light RANSAC pass here is used ONLY to
    discard gross SIFT mismatches — the fitted transform itself is thrown
    away immediately, never reused for anything; only the resulting inlier
    correspondences are kept, as a landmark set that is genuinely
    uncorrelated with whatever transform is later being evaluated against
    it (unlike a held-out subset of the SAME detector's matches, which
    still shares that detector's biases with the fitting data).

    Use this when you want to validate a dense deformation field (L1)
    without spending extra model calls on a refit — the same field
    computed for your main comparison can be sampled at these landmark
    points directly, since the landmarks were never involved in producing
    that field or the affine pre-alignment feeding into it.

    Returns (fixed_pts, moving_pts), each (N, 2) in RAW (pre-affine) image
    coordinates, or (None, None) if too few matches survive.
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(fixed_img, fixed_mask)
    kp2, des2 = sift.detectAndCompute(moving_img, moving_mask)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if len([m, n]) == 2 and m.distance < lowe_ratio * n.distance]
    if len(good) < min_matches:
        return None, None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Outlier removal only — the fitted M is discarded immediately below.
    _, mask = cv2.estimateAffine2D(
        dst, src, method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_thresh, maxIters=3000, confidence=0.99)
    if mask is None or int(mask.sum()) < min_matches:
        return None, None

    idx = np.where(mask.ravel() == 1)[0]
    return src[idx].reshape(-1, 2), dst[idx].reshape(-1, 2)


def print_tre_summary(result, label="", px_size_um=None):
    """Matches the 'Global TRE summary' box format already used in the report."""
    print(f"─── {label} TRE summary ───" + "─" * max(0, 40 - len(label)))
    print(f"n pairs        : {result['n']}")
    if px_size_um is not None:
        print(f"mean  TRE      : {result['mean_px']:.2f} px  = {result['mean_um']:.2f} \u00b5m")
        print(f"median TRE     : {result['median_px']:.2f} px  = {result['median_um']:.2f} \u00b5m")
        print(f"Q3 (75th pct)  : {result['q3_px']:.2f} px  = {result['q3_um']:.2f} \u00b5m")
        print(f"P90            : {result['p90_px']:.2f} px  = {result['p90_um']:.2f} \u00b5m")
        print(f"max   TRE      : {result['max_px']:.2f} px  = {result['max_um']:.2f} \u00b5m")
        print(f"std   TRE      : {result['std_px']:.2f} px  = {result['std_um']:.2f} \u00b5m")
    else:
        print(f"mean  TRE      : {result['mean_px']:.2f} px")
        print(f"median TRE     : {result['median_px']:.2f} px")
        print(f"Q3 (75th pct)  : {result['q3_px']:.2f} px")
        print(f"P90            : {result['p90_px']:.2f} px")
        print(f"max   TRE      : {result['max_px']:.2f} px")
        print(f"std   TRE      : {result['std_px']:.2f} px")
    print("─" * 64)