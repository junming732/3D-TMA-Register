"""
VALIS Registration Pipeline
=================================================

Description:
    This script performs automated batch registration of sequential histology image stacks 
    (TMA Cores) using the VALIS framework. 

    The pipeline handles the full lifecycle: 
    JVM initialization -> Registration -> Visual QC Generation -> Image Warping -> JVM Cleanup.

Key Features:
    1. **Robust Alignment:** Executes feature-based rigid registration.
    2. **Automated QC:** Generates Pairwise Overlay grids (Red/Green) from disk images.
    3. **State Management:** Enforces a "Fresh Run" policy by cleaning previous output.
    4. **Dual Output:** Produces both individual registered OME-TIFFs and a fused volume stack.

Usage:
    python register_core.py --core_name <Core_ID> --channel_idx 0

Configuration (`config.py`):
    - `DATASPACE` (or `RAW_DATA_DIR`): Parent path containing raw image folders.
    - `OUTPUT_DIR`: Parent path where results will be generated.
"""

import os
import sys
import argparse
import shutil
import glob
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tifffile
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
    # Added channel index to match notebook logic for QC plots
    parser.add_argument("--channel_idx", type=int, default=0, help="Channel index to use for QC plots (0=DAPI, 1=CK, etc)")
    return parser.parse_args()

def generate_overlay_plots_from_disk(registered_dir, qc_dir, core_name, channel_idx=0):
    """
    Generates the pairwise overlay (Red=Current, Green=Next) to visualize alignment.
    Strictly follows Notebook logic: Reads files from disk -> Extracts Channel -> Normalizes -> Plots.
    """
    print(f"[INFO] Generating QC Plots from disk using Channel {channel_idx}...")
    
    # Logic from Notebook: Sort files by slice number
    def get_slice_number(filename):
        match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
        return int(match.group(1)) if match else 0

    files = sorted(glob.glob(os.path.join(registered_dir, "*.ome.tiff")), key=get_slice_number)
    
    if len(files) < 2:
        print("[WARN] Not enough files for QC plots.")
        return

    # Setup plot grid
    cols = 4
    rows = int(np.ceil((len(files) - 1) / cols))
    if rows == 0: rows = 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    
    # Ensure axes is always iterable
    if hasattr(axes, 'flatten'):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes] if rows == 1 and cols == 1 else np.array(axes).flatten()

    for idx in range(len(files) - 1):
        f1 = files[idx]
        f2 = files[idx+1]
        
        try:
            # Read specific channel (matches notebook logic)
            img1 = tifffile.imread(f1, key=channel_idx)
            img2 = tifffile.imread(f2, key=channel_idx)
            
            # Notebook Logic: Normalize
            def norm(x):
                p99 = np.percentile(x, 99)
                return np.clip(x / p99, 0, 1) if p99 > 0 else x

            s1 = norm(img1)
            s2 = norm(img2)
            
            # Create Overlay: Red=Slice Z, Green=Slice Z+1
            overlay = np.dstack((s1, s2, np.zeros_like(s1)))
            
            ax = axes_flat[idx]
            ax.imshow(overlay)
            ax.set_title(f"Slice {idx+1} vs {idx+2}", fontsize=8)
            ax.axis('off')
            
        except Exception as e:
            print(f"[WARN] Error plotting slice {idx}: {e}")

    # Hide unused subplots
    for j in range(len(files) - 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    overlay_path = os.path.join(qc_dir, f"{core_name}_overlays_ch{channel_idx}.png")
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print(f"  -> Saved overlays to {overlay_path}")

def create_ome_tiff_stack(src_dir, output_path):
    """
    Manually stacks individual TIFFs into a single OME-TIFF.
    Replaces the deprecated 'stack=True' argument in warp_and_save_slides.
    """
    print(f"[INFO] Creating merged stack at: {output_path}")
    
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
    
    reg_slides_dir = os.path.join(output_dir, "registered_slides")

    print(f"--- Processing {args.core_name} ---")
    
    # --- CRITICAL FIX: Clean FIRST, then Create ---
    # Guarantees a fresh run so plots are generated from new data
    if os.path.exists(output_dir):
        print(f"[INFO] Cleaning old results in {output_dir}...")
        shutil.rmtree(output_dir)
    
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
        
        # --- 5. Warp & Save Individual Slides ---
        print("[INFO] Warping and saving individual OME-TIFFs...")
        registrar.warp_and_save_slides(
            dst_dir=reg_slides_dir,
            crop="reference" # Matched to Notebook logic
        )
        
        # --- 6. QC Plots (Matches Notebook Logic) ---
        # Reads from disk, extracts channel, plots overlay
        generate_overlay_plots_from_disk(reg_slides_dir, qc_dir, args.core_name, args.channel_idx)

        # --- 7. Create Merged Stack ---
        print("[INFO] Creating merged stack...")
        merged_stack_path = os.path.join(output_dir, "merged_stack.ome.tif")
        
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