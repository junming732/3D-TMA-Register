import os
import re
import glob
import logging
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sys
# Get the absolute path of the directory containing this script (registration/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (3D-TMA-Register/)
parent_dir = os.path.dirname(current_dir)
# Add parent directory to sys.path so we can import config
sys.path.append(parent_dir)
import config

matplotlib.use('Agg')

# Configuration Constants
DEFAULT_CHANNEL = 6
MONTAGE_COLUMNS = 5
PERCENTILE_UPPER = 99.5
MINIMUM_SLICES = 2
DPI_RESOLUTION = 150

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y_%m_%d %H:%M:%S'
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize raw pairwise alignment of TMA core slices.')
    parser.add_argument('--core_name', type=str, required=True, help='Target core identifier')
    parser.add_argument('--channel_idx', type=int, default=DEFAULT_CHANNEL, help='Target channel index')
    return parser.parse_args()

def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    return int(match.group(1)) if match else 0 

def load_single_image(file_path):
    img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(img)

def generate_qc_montage(vol, output_folder, channel_idx, prefix, core_name):
    n_slices = vol.shape[0]
    if n_slices < MINIMUM_SLICES: 
        logging.warning("Insufficient slices for montage generation.")
        return

    all_pairs = [(i, i+1) for i in range(n_slices - 1)]
    n_pairs = len(all_pairs)
    n_rows = (n_pairs + MONTAGE_COLUMNS - 1) // MONTAGE_COLUMNS
    
    fig, axes = plt.subplots(n_rows, MONTAGE_COLUMNS, figsize=(4 * MONTAGE_COLUMNS, 3 * n_rows))
    
    if n_rows == 1 and MONTAGE_COLUMNS == 1: 
        axes = np.array([[axes]])
    elif n_rows == 1: 
        axes = axes.reshape(1, -1)
    elif MONTAGE_COLUMNS == 1: 
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()

    logging.info(f"Rendering {n_pairs} visual pairs.")
    for idx, (z1, z2) in enumerate(all_pairs):
        s1 = vol[z1, channel_idx, :, :]
        s2 = vol[z2, channel_idx, :, :]
        
        def normalize_intensity(x):
            p99 = np.percentile(x, PERCENTILE_UPPER)
            if p99 == 0: 
                p99 = 1
            return np.clip(x / p99, 0, 1)
        
        overlay = np.dstack((normalize_intensity(s1), normalize_intensity(s2), np.zeros_like(s1)))
        
        ax = axes_flat[idx]
        ax.imshow(overlay)
        ax.set_title(f"Z{z1} to Z{z2}", fontsize=10, fontweight='bold')
        ax.axis('off')

    for idx in range(len(all_pairs), len(axes_flat)): 
        axes_flat[idx].axis('off')
        
    output_path = os.path.join(output_folder, f"{core_name}_{prefix}_Montage.png")
    plt.suptitle(f'Raw Alignment Verification: {core_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI_RESOLUTION, bbox_inches='tight')
    plt.close(fig) 
    logging.info(f"Montage successfully written out to {output_path}")

def main():
    args = parse_arguments()
    target_core = args.core_name
    channel_idx = args.channel_idx
    
    data_base_path = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_NEW")
    work_output = os.path.join(config.DATASPACE, "QC_PreAlignment")
    input_folder = os.path.join(data_base_path, target_core)
    output_folder = os.path.join(work_output, target_core)

    if not os.path.exists(input_folder):
        logging.error(f"Input directory missing: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"Initiating visualization protocol for {target_core}")

    raw_files = glob.glob(os.path.join(input_folder, "*.ome.tif"))
    file_list = sorted(raw_files, key=get_slice_number)
    total_slices = len(file_list)
    
    if total_slices == 0:
        logging.error("Target directory devoid of valid OME TIFF files.")
        return

    logging.info(f"Detected {total_slices} discrete slices. Executing parallel memory loading.")
    
    loaded_arrays = Parallel(n_jobs=-1)(delayed(load_single_image)(f) for f in file_list)
    
    max_y = max(arr.shape[1] for arr in loaded_arrays)
    max_x = max(arr.shape[2] for arr in loaded_arrays)
    logging.info(f"Establishing spatial conformity to global maximum Y:{max_y} X:{max_x}")

    standardized_arrays = []
    for arr in loaded_arrays:
        c, y, x = arr.shape
        pad_y = max_y - y
        pad_x = max_x - x
        padded_arr = np.pad(arr, ((0, 0), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
        standardized_arrays.append(padded_arr)

    try:
        vol = np.array(standardized_arrays)
        logging.info(f"Volumetric matrix synthesized. Dimensions: {vol.shape}")
    except ValueError as matrix_error:
        logging.error(f"Matrix synthesis failed. Details: {matrix_error}")
        return

    generate_qc_montage(vol, output_folder, channel_idx=channel_idx, prefix="RAW_Overlay", core_name=target_core)

if __name__ == "__main__":
    main()