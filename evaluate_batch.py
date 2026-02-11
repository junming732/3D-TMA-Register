"""
VALIS Batch Evaluation (Bioinformatics 2018 Standards)
======================================================
Logic:
1. Scans all cores in the Registered folder.
2. Calculates Paper Metrics: Jaccard (Overlap), NCC (Correlation), RMSE (Error), Smoothness.
3. OPTIONAL: Loads TRE (Rigid Error) if available (skips if missing).
4. visualizes results: Distribution Boxplots & Stability Heatmaps.

"""

import os
import glob
import re
import numpy as np
import pandas as pd
import tifffile
import cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Image Processing
from skimage.metrics import normalized_mutual_information as nmi_func
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from sklearn.metrics import normalized_mutual_info_score

try:
    import config
except ImportError:
    print("[CRITICAL] config.py not found.")
    import sys; sys.exit(1)

matplotlib.use('Agg')

# --- Tuning ---
DOWNSAMPLE = 4  # 4x downsampling (approx 1024px) is optimal for batch metrics

def get_slice_number(filename):
    """Extracts TMA slice number for sorting (e.g., TMA_10 vs TMA_2)."""
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0

def load_tre_metrics(qc_dir):
    """
    Loads TRE from CSV if it exists. 
    Returns empty dict if missing (Graceful degradation).
    """
    csv_path = glob.glob(os.path.join(qc_dir, "*_rigid_metrics.csv"))
    if not csv_path:
        return {}
    
    try:
        df = pd.read_csv(csv_path[0])
        return pd.Series(df["Rigid Error (px)"].values, index=df["Image"]).to_dict()
    except Exception:
        return {}

def calc_jaccard(im1, im2):
    """Paper Metric: Jaccard Index (Overlap)."""
    try:
        t1 = threshold_otsu(im1)
        t2 = threshold_otsu(im2)
        m1 = im1 > t1
        m2 = im2 > t2
        
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        
        return intersection / union if union > 0 else 0
    except:
        return 0

def calc_pixel_metrics(im1, im2):
    """Paper Metrics: RMSE and NCC (on tissue area only)."""
    try:
        if im1.max() == 0: return np.nan, np.nan
        
        try:
            thresh = threshold_otsu(im1)
        except ValueError:
            return np.nan, np.nan

        mask = im1 > thresh
        
        p1 = im1[mask].ravel()
        p2 = im2[mask].ravel()
        
        if len(p1) < 2 or len(p2) < 2:
            return np.nan, np.nan

        # RMSE
        mse = np.mean((p1 - p2) ** 2)
        rmse = np.sqrt(mse)
        
        # NCC (Correlation)
        # Check standard deviation first to avoid RuntimeWarning
        std1 = np.std(p1)
        std2 = np.std(p2)
        
        if std1 == 0 or std2 == 0:
            ncc = 0 # Constant signal = 0 correlation
        else:
            ncc = np.corrcoef(p1, p2)[0, 1]
        
        return rmse, ncc
    except Exception:
        return np.nan, np.nan

def calc_smoothness(im1):
    """Paper Metric: Reconstruction Smoothness (GLCM Contrast)."""
    try:
        if im1.max() == 0: return 0
        
        # 1. Normalize to 0.0 - 1.0 float
        im_norm = im1.astype(float) / im1.max()
        
        # 2. Scale to 0 - 15 (16 levels) and cast to uint8
        # This guarantees values are strictly 0-15 regardless of input range
        img_q = (im_norm * 15).astype(np.uint8)
        
        # 3. GLCM Calculation
        g = graycomatrix(img_q, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
        contrast = graycoprops(g, 'contrast')[0, 0]
        return contrast
        
    except Exception as e:
        # print(f"[DEBUG] Smoothness error: {e}")
        return np.nan

def plot_batch_analytics(df_detailed, df_summary, output_dir):
    """Generates visual analysis plots."""
    print("[INFO] Generating Analytics Plots...")
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution Boxplots (Consistency Check)
    metrics = ["Jaccard", "NCC", "RMSE", "Smoothness"]
    if not df_detailed["TRE_px"].isna().all():
        metrics.insert(0, "TRE_px")
        
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1: axes = [axes]
    
    for i, metric in enumerate(metrics):
        sns.boxplot(data=df_detailed, x="Core", y=metric, ax=axes[i], palette="viridis")
        axes[i].set_title(f"Distribution of {metric} per Core")
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel("")
        
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Batch_Analytics_Distributions.png"), dpi=150)
    plt.close()
    
    # 2. Heatmap of Stability (Jaccard)
    # Pivot: Index=Core, Columns=Pair_Index, Values=Jaccard
    # Identify specific bad slices
    try:
        df_detailed["Pair_Idx"] = df_detailed["Pair"].apply(lambda x: int(x.split('-')[0]))
        pivot_jaccard = df_detailed.pivot_table(index="Core", columns="Pair_Idx", values="Jaccard")
        
        plt.figure(figsize=(15, max(4, len(df_summary)*0.5)))
        sns.heatmap(pivot_jaccard, cmap="RdYlGn", vmin=0, vmax=1, linewidths=.5, cbar_kws={'label': 'Jaccard Index'})
        plt.title("Registration Stability Heatmap (Jaccard Index)")
        plt.xlabel("Slice Pair Index")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Batch_Analytics_Heatmap.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not generate heatmap: {e}")

def process_core(core_path):
    """Analyzes a single core."""
    core_id = os.path.basename(core_path)
    reg_slides_dir = os.path.join(core_path, "registered_slides")
    qc_dir = os.path.join(core_path, "QC")
    
    if not os.path.exists(reg_slides_dir):
        return None

    files = sorted(glob.glob(os.path.join(reg_slides_dir, "*.ome.tiff")), key=get_slice_number)
    if len(files) < 2:
        return None

    # Load TRE (Returns empty dict if missing -> Safe)
    tre_map = load_tre_metrics(qc_dir)

    core_results = []

    for i in range(len(files) - 1):
        f1 = files[i]
        f2 = files[i+1]
        name1 = os.path.basename(f1)
        
        try:
            im1 = tifffile.imread(f1, key=0)[::DOWNSAMPLE, ::DOWNSAMPLE]
            im2 = tifffile.imread(f2, key=0)[::DOWNSAMPLE, ::DOWNSAMPLE]
            
            if im1.shape != im2.shape:
                 im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

            jaccard = calc_jaccard(im1, im2)
            rmse, ncc = calc_pixel_metrics(im1, im2)
            smoothness = calc_smoothness(im1)
            nmi_val = normalized_mutual_info_score(im1.ravel(), im2.ravel())

            # Soft fuzzy match for TRE
            tre = np.nan
            if tre_map:
                for key, val in tre_map.items():
                    if name1 in str(key): 
                        tre = val
                        break

            core_results.append({
                "Core": core_id,
                "Pair": f"{i+1}-{i+2}",
                "TRE_px": tre,
                "Jaccard": jaccard,
                "RMSE": rmse,
                "NCC": ncc,
                "NMI": nmi_val,
                "Smoothness": smoothness
            })
        except Exception as e:
            print(f"[WARN] Error pair {i} in {core_id}: {e}")

    return pd.DataFrame(core_results)

def main():
    ROOT_DIR = os.path.join(config.DATASPACE, "VALIS_Registered")
    all_cores = sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])
    
    all_data = []
    print(f"--- Starting Benchmark of {len(all_cores)} Cores ---\n")

    for core in all_cores:
        print(f"Processing {core}...", end="\r")
        df_core = process_core(os.path.join(ROOT_DIR, core))
        if df_core is not None:
            all_data.append(df_core)
        else:
            print(f"\n[SKIP] {core} (Missing data/slides)")

    if not all_data:
        print("No valid data found.")
        return

    df_detailed = pd.concat(all_data, ignore_index=True)
    
    # Summary
    df_summary = df_detailed.groupby("Core").agg({
        "TRE_px": lambda x: x.mean() if not x.isna().all() else np.nan,
        "Jaccard": "mean",
        "RMSE": "mean",
        "NCC": "mean",
        "NMI": "mean",
        "Smoothness": "mean"
    }).reset_index()

    # Ranking Score (Higher Jaccard/NCC/NMI is better)
    df_summary["Rank_Score"] = (
        (df_summary["Jaccard"] * 0.3) + 
        (df_summary["NCC"] * 0.3) + 
        (df_summary["NMI"] * 0.4)
    )
    df_summary = df_summary.sort_values("Rank_Score", ascending=False)

    # Save CSVs
    out_dir = config.DATASPACE
    df_detailed.to_csv(os.path.join(out_dir, "Batch_EVAL_Detailed.csv"), index=False)
    df_summary.to_csv(os.path.join(out_dir, "Batch_EVAL_Summary.csv"), index=False)
    
    # Generate Plots
    plot_batch_analytics(df_detailed, df_summary, out_dir)
    
    print(f"\n\n[SUCCESS] Reports & Plots saved to {out_dir}")
    print("\nTop 3 Cores by Quality:")
    print(df_summary[["Core", "Jaccard", "Rank_Score"]].head(3))

if __name__ == "__main__":
    main()