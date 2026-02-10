"""
VALIS Registration Pipeline
=================================================

Description:
    This script performs automated batch registration of sequential histology image stacks 
    (TMA Cores) using the VALIS framework. 

    The pipeline handles the full lifecycle: 
    JVM initialization -> Registration -> Error Metric Calculation -> 
    Visual QC Generation -> Image Warping -> JVM Cleanup.

Key Features:
    1. **Robust Alignment:** Executes feature-based rigid registration (default: BRISK) to align 
       serial tissue slices.
    2. **Automated QC:** Generates "Transparency Reports" (CSV metrics, Error Barplots, and 
       Pairwise Overlay grids) allowing for rapid pass/fail assessment without opening images.
    3. **State Management:** Enforces a "Fresh Run" policy by cleaning previous output directories 
       to ensure metrics strictly reflect the current execution.
    4. **Dual Output:** Produces both individual registered OME-TIFFs and a fused, multi-channel 
       volume stack.

Usage:
    python register_core.py --core_name <Core_ID>

    Example:
        python register_core.py --core_name Core11

Arguments:
    --core_name (str): The specific identifier for the core folder (e.g., "Core11"). 
                       This name is appended to the base paths defined in config.py to locate data.

Configuration (`config.py`):
    Requires a local `config.py` module defining the global storage architecture:
    - `DATASPACE` (or `RAW_DATA_DIR`): Parent path containing raw image folders.
    - `OUTPUT_DIR`: Parent path where results will be generated.

Output Structure:
    <OUTPUT_DIR>/<Core_ID>/
    ├── registered_slides/   # Individual aligned OME-TIFF images
    ├── merged_stack/        # Single multi-channel OME-TIFF (Reconstructed Volume)
    └── QC/                  # Quality Control Artifacts
        ├── *_rigid_metrics.csv  # Raw numerical registration error (TRE)
        ├── *_rigid_plot.png     # Visualization of error per slice (outlier detection)
        └── *_overlays.png       # Grid of Red/Green pairwise overlays for visual verification
"""

import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tifffile  # Added for manual stacking
from valis import registration

# --- 1. Import Configuration ---
try:
    import config
except ImportError:
    print("[CRITICAL] Could not import 'config.py'. Ensure it is in the same folder.")
    sys.exit(1)

# Set non-interactive backend for server/HPC usage
matplotlib.use('Agg') 

def parse_args():
    parser = argparse.ArgumentParser(description="Register a single core using VALIS")
    parser.add_argument("--core_name", type=str, required=True, help="Name of the core folder (e.g., Core11)")
    return parser.parse_args()

def save_qc_report(registrar, qc_dir, core_name):
    """Saves registration error metrics (CSV + Barplot)."""
    print(f"[INFO] Generating QC transparency report in {qc_dir}...")
    
    if not hasattr(registrar, 'rigid_registration_errors') or not registrar.rigid_registration_errors:
        print("[WARNING] No rigid registration errors found (Calculations might have been skipped).")
        return

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(registrar.rigid_registration_errors, orient='index')
    df.columns = ["Rigid Error (px)"]
    df.index.name = "Image"
    
    # 1. Save Raw Data
    csv_path = os.path.join(qc_dir, f"{core_name}_rigid_metrics.csv")
    df.to_csv(csv_path)
    
    # 2. Save Visualization
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df.index, y=df["Rigid Error (px)"], color='#4c72b0')
    mean_err = df["Rigid Error (px)"].mean()
    plt.axhline(y=mean_err, color='r', linestyle='--', label=f'Mean: {mean_err:.2f} px')
    plt.xticks(rotation=90, fontsize=8)
    plt.title(f"Rigid Registration Quality: {core_name}")
    plt.tight_layout()
    
    plot_path = os.path.join(qc_dir, f"{core_name}_rigid_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  -> Saved rigid QC plot to {plot_path}")

def generate_overlay_plots(registrar, qc_dir, core_name):
    """
    Generates the pairwise overlay (Red=Current, Green=Next) to visualize alignment.
    This mimics the visualization cell from your notebook.
    """
    print("[INFO] Generating Pairwise Overlay plots...")
    
    # FIX: Use slide_dict.values() because .get_img_list() does not exist in Valis object
    slides = list(registrar.slide_dict.values())
    n_slides = len(slides)
    
    if n_slides < 2:
        return

    # Setup plot grid
    cols = 4
    rows = int(np.ceil((n_slides - 1) / cols))
    if rows == 0: rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    
    # Ensure axes is always iterable
    if hasattr(axes, 'flatten'):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes] if rows == 1 and cols == 1 else np.array(axes).flatten()

    for idx in range(n_slides - 1):
        # Get current and next slide
        s_obj1 = slides[idx]
        s_obj2 = slides[idx + 1]
        
        # Access the low-res images used for registration for speed
        img1 = s_obj1.image
        img2 = s_obj2.image
        
        # Simple normalization for display
        def norm(x):
            p99 = np.percentile(x, 99)
            return np.clip(x / p99, 0, 1) if p99 > 0 else x

        s1 = norm(img1)
        s2 = norm(img2)

        # Create Overlay: Red=img1, Green=img2
        # Ensure dimensions match
        if s1.shape[:2] != s2.shape[:2]:
            h, w = s1.shape[:2]
            s2 = cv2.resize(s2, (w, h))
        
        # Handle 3D (RGB) vs 2D (Grayscale)
        if len(s1.shape) == 3: s1 = np.mean(s1, axis=2)
        if len(s2.shape) == 3: s2 = np.mean(s2, axis=2)

        overlay = np.dstack((s1, s2, np.zeros_like(s1)))

        if idx < len(axes_flat):
            ax = axes_flat[idx]
            ax.imshow(overlay)
            ax.set_title(f"{s_obj1.name} \nvs Next", fontsize=8)
            ax.axis('off')

    # Hide unused subplots
    for j in range(n_slides - 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    overlay_path = os.path.join(qc_dir, f"{core_name}_overlays.png")
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print(f"  -> Saved overlays to {overlay_path}")

def create_ome_tiff_stack(src_dir, output_path):
    """
    Manually stacks individual TIFFs into a single OME-TIFF.
    Replaces the deprecated 'stack=True' argument in warp_and_save_slides.
    """
    print(f"[INFO] Creating merged stack at: {output_path}")
    
    # Find all .tif/.tiff files
    files = sorted([
        os.path.join(src_dir, f) for f in os.listdir(src_dir) 
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    
    if not files:
        print("[WARN] No registered slides found to stack.")
        return

    try:
        # Use tifffile to write a BigTIFF stack
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            for i, fpath in enumerate(files):
                img = tifffile.imread(fpath)
                tif.write(img, description=f"Slice_{i}", metadata={'axes': 'YX'})
                
        print(f"[SUCCESS] Merged stack created: {len(files)} slices.")
        
    except Exception as e:
        print(f"[ERROR] Could not create merged stack: {e}")

def main():
    args = parse_args()
    
    # --- 2. Construct Paths ---
    DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_NEW")
    input_dir = os.path.join(DATA_BASE_PATH, args.core_name)

    WORK_OUTPUT = os.path.join(config.DATASPACE, "Registered")
    output_dir = os.path.join(WORK_OUTPUT, args.core_name)
    qc_dir = os.path.join(output_dir, "QC")
    
    # NEW: Define specific path for registered slides for use in stacking
    reg_slides_dir = os.path.join(output_dir, "registered_slides")

    print(f"--- Processing {args.core_name} ---")
    
    # --- CRITICAL FIX: Clean FIRST, then Create ---
    if os.path.exists(output_dir):
        print(f"[INFO] Cleaning old results in {output_dir} to force fresh metrics...")
        shutil.rmtree(output_dir)
    
    # Now create the folders (including QC)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    # Check if input exists
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    # --- 3. Initialize VALIS ---
    try:
        registrar = registration.Valis(
            src_dir=input_dir,
            dst_dir=output_dir,
            imgs_ordered=True
        )

        # --- 4. Registration ---
        print("[INFO] Starting Registration Pipeline...")
        registrar.register() 
        
        # --- 5. Transparency / QC ---
        save_qc_report(registrar, qc_dir, args.core_name)
        
        # Add Overlay Plot Generation
        try:
            generate_overlay_plots(registrar, qc_dir, args.core_name)
        except Exception as e:
            print(f"[WARN] Failed to generate overlay plots: {e}")

        # --- 6. Warp & Save Individual Slides ---
        print("[INFO] Warping and saving individual OME-TIFFs...")
        registrar.warp_and_save_slides(
            dst_dir=reg_slides_dir,
            crop="overlap" 
        )

        # --- 7. Create Merged Stack (FIXED) ---
        # Replaced the crashing 'stack=True' call with manual stacking function
        print("[INFO] Creating merged stack...")
        merged_stack_path = os.path.join(output_dir, "merged_stack.ome.tif")
        
        # Create directory for merged stack if needed (or save in root of output)
        os.makedirs(os.path.dirname(merged_stack_path), exist_ok=True)
        
        create_ome_tiff_stack(reg_slides_dir, merged_stack_path)

        print(f"[SUCCESS] Core {args.core_name} processed successfully.")

    except Exception as e:
        print(f"[FAILURE] Error processing {args.core_name}: {e}")
        raise e
        
    finally:
        print("[INFO] Killing JVM...")
        registration.kill_jvm()

if __name__ == "__main__":
    main()