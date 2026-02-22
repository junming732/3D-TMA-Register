"""
Quantitatively compare alignment quality at three stages:
  1. Raw (unregistered) slices
  2. After intensity-based registration  (Amoeba / MeanSquares)
  3. After feature-based registration    (BRISK + AKAZE fallback)

Metrics computed per consecutive slice pair (Z_i vs Z_{i+1}):
  - MSE          Mean Squared Error                (lower = better)
  - RMSE         Root MSE                          (lower = better)
  - NCC          Normalised Cross-Correlation      (higher = better, max 1.0)
  - MI           Mutual Information (Shannon)      (higher = better)
  - NMI          Normalised MI                     (higher = better, max 2.0)
  - SSIM         Structural Similarity Index       (higher = better, max 1.0)
  - PSNR         Peak Signal-to-Noise Ratio (dB)   (higher = better)
  - GradCorr     Gradient magnitude correlation    (higher = better)

All metrics are computed on the CK channel (default idx 6) at FULL resolution
and at a 4× downsampled version (for speed sanity-check).

Usage
-----
python registration/evaluate_registration.py --core_name Core_01
python registration/evaluate_registration.py --core_name Core_01 --channel 0
python registration/evaluate_registration.py --core_name Core_01 --no_downsample_check
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tifffile
import glob
import re

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# ── path setup (same as other scripts) ──────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

# ── scipy / skimage (available in the same venv) ────────────────────────────
from scipy.ndimage import sobel
from scipy.stats  import pearsonr
try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[WARNING] scikit-image not found — SSIM will be NaN.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate registration quality at three stages.")
parser.add_argument('--core_name',          type=str,  required=True, help='e.g. Core_01')
parser.add_argument('--channel',            type=int,  default=6,     help='Channel index for metrics (default: 6 = CK)')
parser.add_argument('--no_downsample_check',action='store_true',      help='Skip the 4× downsampled metric pass')
parser.add_argument('--bins_mi',            type=int,  default=64,    help='Histogram bins for MI computation (default: 64)')
args = parser.parse_args()

TARGET_CORE   = args.core_name
CHANNEL_IDX   = args.channel
BINS_MI       = args.bins_mi
DO_DS_CHECK   = not args.no_downsample_check

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
RAW_INPUT_FOLDER    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_NEW",         TARGET_CORE)
AMOEBA_VOLUME_PATH  = os.path.join(config.DATASPACE, "Amoeba_Registered_rigid_geometry", TARGET_CORE,
                                   f"{TARGET_CORE}_Amoeba_Aligned.ome.tif")
FEATURE_VOLUME_PATH = os.path.join(config.DATASPACE, "Feature_Registered_rigid_geometry", TARGET_CORE,
                                   f"{TARGET_CORE}_Feature_Aligned.ome.tif")
OUTPUT_FOLDER       = os.path.join(config.DATASPACE, "Registration_Evaluation",       TARGET_CORE)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print(f"  Registration Evaluation: {TARGET_CORE}")
print(f"  Channel: {CHANNEL_IDX}  |  MI bins: {BINS_MI}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: load raw slices sorted by slice number
# ─────────────────────────────────────────────────────────────────────────────
def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def load_raw_volume(folder: str, channel: int) -> np.ndarray:
    """
    Load individual OME-TIFFs from the raw input folder and stack into
    a (Z, H, W) array for the requested channel.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.ome.tif")), key=get_slice_number)
    if not files:
        raise FileNotFoundError(f"No OME-TIFF files found in: {folder}")
    slices = []
    for f in files:
        img = tifffile.imread(f)             # shape may be (C, H, W) or (H, W, C) or (H, W)
        # Normalise to (C, H, W)
        if img.ndim == 2:
            img = img[np.newaxis]            # single-channel
        elif img.ndim == 3 and img.shape[-1] < img.shape[0]:
            img = np.moveaxis(img, -1, 0)   # (H, W, C) → (C, H, W)
        slices.append(img[channel])
    return np.stack(slices, axis=0).astype(np.float32)   # (Z, H, W)


def load_registered_volume(path: str, channel: int) -> np.ndarray:
    """Load a (Z, C, H, W) OME-TIFF and return the (Z, H, W) channel slice."""
    vol = tifffile.imread(path)     # expected (Z, C, H, W)
    if vol.ndim == 3:               # (Z, H, W) — single channel saved
        return vol.astype(np.float32)
    if vol.ndim == 4:
        # Axis order: tifffile usually gives (Z, C, H, W) for axes='ZCYX'
        if vol.shape[1] < vol.shape[2]:          # (Z, C, H, W)
            return vol[:, channel].astype(np.float32)
        else:                                     # (Z, H, W, C) fallback
            return vol[:, :, :, channel].astype(np.float32)
    raise ValueError(f"Unexpected volume shape: {vol.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS  (all operate on 2-D float arrays, same spatial size)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(x):
    return float(x) if np.isfinite(x) else float('nan')


def metric_mse(a: np.ndarray, b: np.ndarray) -> float:
    return _safe_float(np.mean((a - b) ** 2))


def metric_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return _safe_float(np.sqrt(np.mean((a - b) ** 2)))


def metric_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised Cross-Correlation via Pearson r."""
    a_f, b_f = a.ravel(), b.ravel()
    if a_f.std() < 1e-6 or b_f.std() < 1e-6:
        return float('nan')
    r, _ = pearsonr(a_f, b_f)
    return _safe_float(r)


def metric_mi(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Shannon Mutual Information via joint histogram."""
    a_n = (a - a.min()) / (a.ptp() + 1e-9)
    b_n = (b - b.min()) / (b.ptp() + 1e-9)
    hist2d, _, _ = np.histogram2d(a_n.ravel(), b_n.ravel(), bins=bins)
    pxy = hist2d / hist2d.sum()
    px  = pxy.sum(axis=1, keepdims=True)
    py  = pxy.sum(axis=0, keepdims=True)
    mask = pxy > 0
    mi = float(np.sum(pxy[mask] * np.log(pxy[mask] / (px * py + 1e-12)[mask])))
    return _safe_float(mi)


def metric_nmi(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Normalised MI = (H(A) + H(B)) / H(A,B)."""
    def entropy(p):
        p = p[p > 0]
        return -float(np.sum(p * np.log(p)))

    a_n = (a - a.min()) / (a.ptp() + 1e-9)
    b_n = (b - b.min()) / (b.ptp() + 1e-9)
    hist2d, _, _ = np.histogram2d(a_n.ravel(), b_n.ravel(), bins=bins)
    pxy = hist2d / hist2d.sum()
    px  = pxy.sum(axis=1)
    py  = pxy.sum(axis=0)
    h_ab = entropy(pxy)
    h_a  = entropy(px)
    h_b  = entropy(py)
    if h_ab < 1e-9:
        return float('nan')
    return _safe_float((h_a + h_b) / h_ab)


def metric_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if not HAS_SKIMAGE:
        return float('nan')
    # SSIM needs values in a consistent range; normalise to [0,1]
    vmax = max(a.max(), b.max(), 1.0)
    return _safe_float(ssim_fn(a / vmax, b / vmax, data_range=1.0))


def metric_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return float('inf')
    data_range = max(a.max(), b.max(), 1.0)
    return _safe_float(20 * np.log10(data_range / np.sqrt(mse)))


def metric_grad_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Correlation of gradient magnitudes.
    Captures edge-level structural alignment independent of intensity offset.
    """
    def grad_mag(x):
        gx = sobel(x, axis=0)
        gy = sobel(x, axis=1)
        return np.hypot(gx, gy)

    ga, gb = grad_mag(a).ravel(), grad_mag(b).ravel()
    if ga.std() < 1e-6 or gb.std() < 1e-6:
        return float('nan')
    r, _ = pearsonr(ga, gb)
    return _safe_float(r)


ALL_METRICS = {
    "MSE":      metric_mse,
    "RMSE":     metric_rmse,
    "NCC":      metric_ncc,
    "MI":       lambda a, b: metric_mi(a, b, bins=BINS_MI),
    "NMI":      lambda a, b: metric_nmi(a, b, bins=BINS_MI),
    "SSIM":     metric_ssim,
    "PSNR":     metric_psnr,
    "GradCorr": metric_grad_corr,
}

# Which metrics are "higher = better" (for delta sign interpretation)
HIGHER_IS_BETTER = {"NCC", "MI", "NMI", "SSIM", "PSNR", "GradCorr"}


# ─────────────────────────────────────────────────────────────────────────────
# CORE EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_volume(vol: np.ndarray, stage_name: str, downsample: int = 1) -> pd.DataFrame:
    """
    Compute all metrics for every consecutive slice pair in vol (Z, H, W).
    downsample > 1 → apply average pooling before metric computation.
    Returns a DataFrame with one row per pair.
    """
    n_slices = vol.shape[0]
    records  = []

    for z in range(n_slices - 1):
        a = vol[z]
        b = vol[z + 1]

        if downsample > 1:
            from skimage.transform import downscale_local_mean
            a = downscale_local_mean(a, (downsample, downsample)).astype(np.float32)
            b = downscale_local_mean(b, (downsample, downsample)).astype(np.float32)

        row = {"Stage": stage_name, "Pair": f"Z{z:02d}-Z{z+1:02d}", "Z_idx": z}
        for name, fn in ALL_METRICS.items():
            row[name] = fn(a, b)
        records.append(row)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/4] Loading volumes...")

# Raw
print(f"  Raw:     {RAW_INPUT_FOLDER}")
try:
    vol_raw = load_raw_volume(RAW_INPUT_FOLDER, CHANNEL_IDX)
    print(f"           shape={vol_raw.shape}")
except Exception as e:
    print(f"  [ERROR] Could not load raw volume: {e}")
    sys.exit(1)

n_slices = vol_raw.shape[0]

# Amoeba
print(f"  Amoeba:  {AMOEBA_VOLUME_PATH}")
try:
    vol_amoeba = load_registered_volume(AMOEBA_VOLUME_PATH, CHANNEL_IDX)
    print(f"           shape={vol_amoeba.shape}")
    HAVE_AMOEBA = True
except Exception as e:
    print(f"  [WARNING] Amoeba volume not available: {e}")
    vol_amoeba = None
    HAVE_AMOEBA = False

# Feature
print(f"  Feature: {FEATURE_VOLUME_PATH}")
try:
    vol_feature = load_registered_volume(FEATURE_VOLUME_PATH, CHANNEL_IDX)
    print(f"           shape={vol_feature.shape}")
    HAVE_FEATURE = True
except Exception as e:
    print(f"  [WARNING] Feature volume not available: {e}")
    vol_feature = None
    HAVE_FEATURE = False

if not HAVE_AMOEBA and not HAVE_FEATURE:
    print("[ERROR] No registered volumes found. Run registration scripts first.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/4] Computing metrics (full resolution)...")
t0 = time.time()

dfs = []

print("  Stage: Raw...")
dfs.append(evaluate_volume(vol_raw, "1_Raw"))

if HAVE_AMOEBA:
    print("  Stage: Amoeba (intensity-based)...")
    dfs.append(evaluate_volume(vol_amoeba, "2_Amoeba"))

if HAVE_FEATURE:
    print("  Stage: Feature (BRISK+AKAZE)...")
    dfs.append(evaluate_volume(vol_feature, "3_Feature"))

df_full = pd.concat(dfs, ignore_index=True)
print(f"  Done in {time.time()-t0:.1f}s")

if DO_DS_CHECK:
    print("\n  Computing metrics (4× downsampled for sanity-check)...")
    t0 = time.time()
    ds_dfs = []
    ds_dfs.append(evaluate_volume(vol_raw,     "1_Raw_DS4",    downsample=4))
    if HAVE_AMOEBA:
        ds_dfs.append(evaluate_volume(vol_amoeba, "2_Amoeba_DS4", downsample=4))
    if HAVE_FEATURE:
        ds_dfs.append(evaluate_volume(vol_feature,"3_Feature_DS4",downsample=4))
    df_ds = pd.concat(ds_dfs, ignore_index=True)
    print(f"  Done in {time.time()-t0:.1f}s")
else:
    df_ds = pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RAW NUMBERS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/4] Saving CSVs...")
full_csv = os.path.join(OUTPUT_FOLDER, "metrics_full_resolution.csv")
df_full.to_csv(full_csv, index=False)
print(f"  Full-res metrics → {full_csv}")

if not df_ds.empty:
    ds_csv = os.path.join(OUTPUT_FOLDER, "metrics_ds4_resolution.csv")
    df_ds.to_csv(ds_csv, index=False)
    print(f"  4× DS metrics   → {ds_csv}")

# ── Summary table: mean ± std per stage ─────────────────────────────────────
metric_cols = list(ALL_METRICS.keys())

summary_rows = []
for stage, grp in df_full.groupby("Stage"):
    row = {"Stage": stage, "N_Pairs": len(grp)}
    for m in metric_cols:
        vals = grp[m].dropna()
        row[f"{m}_mean"] = round(vals.mean(), 5) if len(vals) else float('nan')
        row[f"{m}_std"]  = round(vals.std(),  5) if len(vals) else float('nan')
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUTPUT_FOLDER, "metrics_summary.csv")
df_summary.to_csv(summary_csv, index=False)
print(f"  Summary          → {summary_csv}")

# ── Delta table: improvement at each stage ───────────────────────────────────
print("\n  --- Improvement Summary ---")
stages = df_full["Stage"].unique()

delta_rows = []
for m in metric_cols:
    r = {"Metric": m, "Direction": "↑ higher=better" if m in HIGHER_IS_BETTER else "↓ lower=better"}
    means = {s: df_full[df_full["Stage"] == s][m].mean() for s in stages}
    r["Raw_mean"] = round(means.get("1_Raw", float('nan')), 5)
    if HAVE_AMOEBA:
        r["Amoeba_mean"]  = round(means.get("2_Amoeba", float('nan')), 5)
        delta_a = means.get("2_Amoeba", float('nan')) - means.get("1_Raw", float('nan'))
        sign = 1 if m in HIGHER_IS_BETTER else -1
        r["Delta_Raw→Amoeba"]   = round(delta_a, 5)
        r["Improvement_Raw→Amoeba_%"] = round(sign * delta_a / (abs(means.get("1_Raw", 1)) + 1e-12) * 100, 2)
    if HAVE_FEATURE:
        r["Feature_mean"] = round(means.get("3_Feature", float('nan')), 5)
        delta_f = means.get("3_Feature", float('nan')) - means.get("2_Amoeba" if HAVE_AMOEBA else "1_Raw", float('nan'))
        ref_key = "2_Amoeba" if HAVE_AMOEBA else "1_Raw"
        ref_label = "Amoeba" if HAVE_AMOEBA else "Raw"
        r[f"Delta_{ref_label}→Feature"]         = round(delta_f, 5)
        r[f"Improvement_{ref_label}→Feature_%"] = round(sign * delta_f / (abs(means.get(ref_key, 1)) + 1e-12) * 100, 2)
    delta_rows.append(r)

df_delta = pd.DataFrame(delta_rows)
delta_csv = os.path.join(OUTPUT_FOLDER, "metrics_delta.csv")
df_delta.to_csv(delta_csv, index=False)
print(f"  Delta table      → {delta_csv}")
print()
print(df_delta.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/4] Generating plots...")

stage_labels = df_full["Stage"].unique().tolist()
stage_colors = {"1_Raw": "#d62728", "2_Amoeba": "#1f77b4", "3_Feature": "#2ca02c"}
stage_display = {
    "1_Raw":     "Raw",
    "2_Amoeba":  "Amoeba\n(intensity)",
    "3_Feature": "Feature\n(BRISK+AKAZE)"
}

# ── Plot 1: Per-pair metric traces (one subplot per metric) ──────────────────
n_metrics = len(metric_cols)
fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 3.5 * n_metrics), sharex=False)
if n_metrics == 1:
    axes = [axes]

pair_labels = df_full[df_full["Stage"] == "1_Raw"]["Pair"].tolist()
x_vals = np.arange(len(pair_labels))

for ax, m in zip(axes, metric_cols):
    for stage in stage_labels:
        grp = df_full[df_full["Stage"] == stage].sort_values("Z_idx")
        vals = grp[m].values
        color = stage_colors.get(stage, "grey")
        ax.plot(x_vals[:len(vals)], vals, marker='o', markersize=4,
                label=stage_display.get(stage, stage), color=color, linewidth=1.5)
        ax.fill_between(x_vals[:len(vals)], vals, alpha=0.08, color=color)

    ax.set_title(m, fontsize=12, fontweight='bold')
    ax.set_ylabel(m)
    if len(x_vals) <= 20:
        ax.set_xticks(x_vals)
        ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    direction_text = "↑ higher=better" if m in HIGHER_IS_BETTER else "↓ lower=better"
    ax.text(0.01, 0.97, direction_text, transform=ax.transAxes,
            fontsize=7, va='top', color='grey')

plt.suptitle(f"{TARGET_CORE} | Ch={CHANNEL_IDX} | Per-Pair Metrics Across Registration Stages",
             fontsize=14, fontweight='bold', y=1.001)
plt.tight_layout()
trace_path = os.path.join(OUTPUT_FOLDER, "metrics_per_pair_traces.png")
plt.savefig(trace_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Traces plot    → {trace_path}")


# ── Plot 2: Box plots per metric, grouped by stage ───────────────────────────
n_cols = 4
n_rows = (n_metrics + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes_flat = np.array(axes).flatten()

for idx, m in enumerate(metric_cols):
    ax = axes_flat[idx]
    data_by_stage = []
    labels_clean  = []
    colors_list   = []
    for stage in stage_labels:
        vals = df_full[df_full["Stage"] == stage][m].dropna().values
        if len(vals) > 0:
            data_by_stage.append(vals)
            labels_clean.append(stage_display.get(stage, stage))
            colors_list.append(stage_colors.get(stage, "grey"))

    bp = ax.boxplot(data_by_stage, patch_artist=True, labels=labels_clean,
                    notch=False, showfliers=True, flierprops=dict(marker='.', alpha=0.5))
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title(m, fontweight='bold')
    direction_text = "↑ higher=better" if m in HIGHER_IS_BETTER else "↓ lower=better"
    ax.set_xlabel(direction_text, fontsize=7, color='grey')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

for idx in range(n_metrics, len(axes_flat)):
    axes_flat[idx].axis('off')

plt.suptitle(f"{TARGET_CORE} | Ch={CHANNEL_IDX} | Metric Distributions by Stage",
             fontsize=14, fontweight='bold')
plt.tight_layout()
box_path = os.path.join(OUTPUT_FOLDER, "metrics_boxplots.png")
plt.savefig(box_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Box plots      → {box_path}")


# ── Plot 3: Summary bar chart (mean metric, grouped by stage) ────────────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes_flat = np.array(axes).flatten()
x_pos = np.arange(len(stage_labels))

for idx, m in enumerate(metric_cols):
    ax = axes_flat[idx]
    means  = [df_full[df_full["Stage"] == s][m].mean() for s in stage_labels]
    stds   = [df_full[df_full["Stage"] == s][m].std()  for s in stage_labels]
    colors = [stage_colors.get(s, 'grey') for s in stage_labels]

    bars = ax.bar(x_pos, means, yerr=stds, capsize=4, color=colors, alpha=0.75, edgecolor='black')
    for bar, val in zip(bars, means):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.3g}", ha='center', va='bottom', fontsize=7)

    ax.set_title(m, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([stage_display.get(s, s) for s in stage_labels], fontsize=7)
    direction_text = "↑ higher=better" if m in HIGHER_IS_BETTER else "↓ lower=better"
    ax.set_xlabel(direction_text, fontsize=7, color='grey')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

for idx in range(n_metrics, len(axes_flat)):
    axes_flat[idx].axis('off')

plt.suptitle(f"{TARGET_CORE} | Ch={CHANNEL_IDX} | Mean Metrics (±SD) by Stage",
             fontsize=14, fontweight='bold')
plt.tight_layout()
bar_path = os.path.join(OUTPUT_FOLDER, "metrics_bar_summary.png")
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Bar summary    → {bar_path}")


# ── Plot 4: Delta heatmap (% improvement relative to Raw) ───────────────────
improvement_data = {}
for m in metric_cols:
    sign = 1 if m in HIGHER_IS_BETTER else -1
    raw_mean = df_full[df_full["Stage"] == "1_Raw"][m].mean()
    for stage in stage_labels:
        if stage == "1_Raw":
            continue
        s_mean = df_full[df_full["Stage"] == stage][m].mean()
        pct = sign * (s_mean - raw_mean) / (abs(raw_mean) + 1e-12) * 100
        improvement_data.setdefault(m, {})[stage] = round(pct, 2)

if improvement_data:
    non_raw_stages = [s for s in stage_labels if s != "1_Raw"]
    heat_matrix = np.array([
        [improvement_data[m].get(s, float('nan')) for s in non_raw_stages]
        for m in metric_cols
    ])

    fig, ax = plt.subplots(figsize=(max(4, 2 * len(non_raw_stages)), max(5, 0.6 * n_metrics)))
    im = ax.imshow(heat_matrix, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
    plt.colorbar(im, ax=ax, label='% improvement vs Raw\n(sign-corrected so green=better)')

    ax.set_xticks(range(len(non_raw_stages)))
    ax.set_xticklabels([stage_display.get(s, s).replace('\n', ' ') for s in non_raw_stages])
    ax.set_yticks(range(len(metric_cols)))
    ax.set_yticklabels(metric_cols)

    for i in range(len(metric_cols)):
        for j in range(len(non_raw_stages)):
            val = heat_matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}%", ha='center', va='center',
                        fontsize=9, color='black', fontweight='bold')

    ax.set_title(f"{TARGET_CORE} | % Improvement vs Raw (green = better)",
                 fontweight='bold', pad=12)
    plt.tight_layout()
    heat_path = os.path.join(OUTPUT_FOLDER, "metrics_improvement_heatmap.png")
    plt.savefig(heat_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap        → {heat_path}")


# ── Plot 5: Visual overlay comparison (centre pair, all three stages) ────────
center_z = n_slices // 2
if center_z + 1 < n_slices:
    stages_avail = [("Raw", vol_raw)]
    if HAVE_AMOEBA:
        stages_avail.append(("Amoeba", vol_amoeba))
    if HAVE_FEATURE:
        stages_avail.append(("Feature\n(BRISK+AKAZE)", vol_feature))

    fig, axes = plt.subplots(1, len(stages_avail), figsize=(6 * len(stages_avail), 5))
    if len(stages_avail) == 1:
        axes = [axes]

    def make_overlay(s1, s2):
        def norm(x):
            p99 = np.percentile(x, 99.5)
            return np.clip(x / (p99 if p99 > 0 else 1), 0, 1)
        return np.dstack((norm(s1), norm(s2), np.zeros_like(s1)))

    for ax, (label, vol) in zip(axes, stages_avail):
        s1 = vol[center_z].astype(np.float32)
        s2 = vol[center_z + 1].astype(np.float32)
        overlay = make_overlay(s1, s2)
        mse_val = metric_mse(s1, s2)
        ncc_val = metric_ncc(s1, s2)
        ax.imshow(overlay)
        ax.set_title(f"{label}\nMSE={mse_val:.1f}  NCC={ncc_val:.4f}", fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f"{TARGET_CORE} | Centre Pair Z{center_z}–Z{center_z+1} | Green=Z{center_z}, Red=Z{center_z+1}",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    overlay_path = os.path.join(OUTPUT_FOLDER, "visual_overlay_centre_pair.png")
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Overlay visual → {overlay_path}")


print("\n" + "=" * 60)
print(f"  Evaluation complete for {TARGET_CORE}")
print(f"  Output folder: {OUTPUT_FOLDER}")
print("=" * 60)