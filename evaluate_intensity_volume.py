"""
Intensity-Based Batch Evaluation 
====================================================
Adapts the Bioinformatics 2018 metrics for 3D integrated volumes.
Logic:
1. Loads the full 3D stack/volume file.
2. Slices it along the Z-axis.
3. Calculates pairwise metrics (Jaccard, RMSE, NCC, Smoothness) between Z and Z+1.

Usage:
    python evaluate_intensity_volume.py
"""

import os
import glob
import numpy as np
import pandas as pd
import tifffile
import SimpleITK as sitk
import cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Image Processing
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
DOWNSAMPLE = 4 

def load_volume(filepath):
    """Loads a 3D volume from .tiff"""
    try:
        # Tifffile handles multi-page tiffs efficiently
        vol = tifffile.imread(filepath)
        # Ensure shape is (Z, Y, X)
        if vol.ndim == 2: return np.array([vol]) # Single slice
        return vol
        
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return None

def calc_jaccard(im1, im2):
    try:
        if im1.max() == 0 or im2.max() == 0: return 0
        try:
            #TODO:  integrate theresholding with preprocessing
            t1 = threshold_otsu(im1)
            t2 = threshold_otsu(im2)
        except ValueError: return 0
        m1 = im1 > t1
        m2 = im2 > t2
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return intersection / union if union > 0 else 0
    except: return 0

def calc_pixel_metrics(im1, im2):
    try:
        if im1.max() == 0: return np.nan, np.nan
        #TODO:  integrate theresholding with preprocessing
        try: thresh = threshold_otsu(im1)
        except: return np.nan, np.nan
        
        mask = im1 > thresh
        p1 = im1[mask].ravel()
        p2 = im2[mask].ravel()
        
        if len(p1) < 2 or len(p2) < 2: return np.nan, np.nan

        mse = np.mean((p1 - p2) ** 2)
        rmse = np.sqrt(mse)
        
        std1, std2 = np.std(p1), np.std(p2)
        ncc = 0 if (std1 == 0 or std2 == 0) else np.corrcoef(p1, p2)[0, 1]
        
        return rmse, ncc
    except: return np.nan, np.nan

def calc_smoothness(im1):
    try:
        if im1.max() == 0: return 0
        im_norm = im1.astype(float) / im1.max()
        img_q = (im_norm * 15).astype(np.uint8)
        g = graycomatrix(img_q, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
        return graycoprops(g, 'contrast')[0, 0]
    except: return np.nan

def process_volume_file(filepath, core_id):
    """Slices the 3D volume and calculates pairwise stats."""
    print(f"Loading {core_id}...", end="\r")
    
    vol = load_volume(filepath)
    if vol is None or len(vol) < 2:
        return None

    # Handle if volume has channels (Z, C, Y, X) -> Extract Ch0
    if vol.ndim == 4:
        # Assuming channel is 2nd dim (Z, C, Y, X) or last (Z, Y, X, C)
        # Heuristic: Smallest dim is usually channel
        if vol.shape[1] < 10: vol = vol[:, 0, :, :] # Pick Ch0
        elif vol.shape[3] < 10: vol = vol[:, :, :, 0]

    results = []
    
    # Iterate Z-slices
    for z in range(len(vol) - 1):
        try:
            im1 = vol[z][::DOWNSAMPLE, ::DOWNSAMPLE]
            im2 = vol[z+1][::DOWNSAMPLE, ::DOWNSAMPLE]
            
            # Simple resize safety
            if im1.shape != im2.shape:
                im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

            jaccard = calc_jaccard(im1, im2)
            rmse, ncc = calc_pixel_metrics(im1, im2)
            smoothness = calc_smoothness(im1)
            nmi_val = normalized_mutual_info_score(im1.ravel(), im2.ravel())

            results.append({
                "Core": core_id,
                "Pair": f"{z+1}-{z+2}",
                "Jaccard": jaccard,
                "RMSE": rmse,
                "NCC": ncc,
                "NMI": nmi_val,
                "Smoothness": smoothness
            })
        except Exception as e:
            pass

    return pd.DataFrame(results)

def plot_consistent_analytics(df_detailed, df_summary, output_dir):
    """
    Generates exact same plots as VALIS evaluation:
    1. Distribution Boxplots
    2. Stability Heatmap
    """
    print("\n[INFO] Generating Consistent Analytics Plots...")
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution Boxplots
    metrics = ["Jaccard", "NCC", "RMSE", "Smoothness"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1: axes = [axes]
    
    for i, metric in enumerate(metrics):
        clean_data = df_detailed.dropna(subset=[metric])
        if not clean_data.empty:
            sns.boxplot(data=clean_data, x="Core", y=metric, ax=axes[i], palette="viridis")
            axes[i].set_title(f"Distribution of {metric} per Core")
            axes[i].set_ylabel(metric)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Intensity_Analytics_Distributions_DAPI.png"), dpi=150)
    plt.close()
    
    # 2. Heatmap
    try:
        # Extract Pair Index (1 from "1-2")
        df_detailed["Pair_Idx"] = df_detailed["Pair"].apply(lambda x: int(x.split('-')[0]) if '-' in str(x) else 0)
        pivot_jaccard = df_detailed.pivot_table(index="Core", columns="Pair_Idx", values="Jaccard")
        
        plt.figure(figsize=(15, max(4, len(df_summary)*0.5)))
        sns.heatmap(pivot_jaccard, cmap="RdYlGn", vmin=0, vmax=1, linewidths=.5, cbar_kws={'label': 'Jaccard Index'})
        plt.title("Registration Stability Heatmap (Jaccard Index)")
        plt.xlabel("Slice Pair Index")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Intensity_Analytics_Heatmap_DAPI.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] Heatmap generation failed: {e}")

def main():
    # DIRECTLY TARGETING YOUR FOLDER STRUCTURE
    # /data3/junming/3D-TMA-Register/Intensity_Registered/
    ROOT_DIR = os.path.join(config.DATASPACE, "Intensity_Registered_DAPI")
    
    # Find specific files: Core_XX/Core_XX_CenterOut_Aligned_CK.ome.tif
    search_pattern = os.path.join(ROOT_DIR, "**", "*Aligned_CK.ome.tif")
    files = sorted(glob.glob(search_pattern, recursive=True))
    
    all_data = []
    print(f"--- Found {len(files)} Intensity Volumes ---\n")

    for fp in files:
        # Extract Core ID (Core_01, Core_11, etc.)
        # Assumes folder name is the Core ID
        core_id = os.path.basename(os.path.dirname(fp))
        
        df = process_volume_file(fp, core_id)
        if df is not None:
            all_data.append(df)
        else:
            print(f"[SKIP] {core_id} (Load failed)")

    if not all_data:
        print("No data found.")
        return

    df_detailed = pd.concat(all_data, ignore_index=True)
    
    # Summary
    df_summary = df_detailed.groupby("Core").agg({
        "Jaccard": "mean", "RMSE": "mean", "NCC": "mean", 
        "NMI": "mean", "Smoothness": "mean"
    }).reset_index()

    # Rank Score: Higher is Better
    # Note: Smoothness (Contrast) is better when LOWER.
    # We invert smoothness for the ranking (1/Smoothness or -Smoothness)
    # Simple Rank: 40% Jaccard + 40% NCC + 20% NMI
    df_summary["Rank_Score"] = (df_summary["Jaccard"]*0.4 + df_summary["NCC"]*0.4 + df_summary["NMI"]*0.2)
    df_summary = df_summary.sort_values("Rank_Score", ascending=False)

    # Save
    out_dir = config.DATASPACE
    df_detailed.to_csv(os.path.join(out_dir, "Intensity_DAPI_EVAL_Detailed.csv"), index=False)
    df_summary.to_csv(os.path.join(out_dir, "Intensity_DAPI_EVAL_Summary.csv"), index=False)
    
    plot_consistent_analytics(df_detailed, df_summary, out_dir)
    
    print(f"\n\n[SUCCESS] Reports saved to {out_dir}")
    print("\nTop 3 Intensity Cores:")
    print(df_summary[["Core", "Jaccard", "Rank_Score"]].head(3))

if __name__ == "__main__":
    main()