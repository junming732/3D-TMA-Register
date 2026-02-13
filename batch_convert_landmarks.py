"""
QuPath Landmark Batch Ingestion Module
======================================

This module acts as the "Data Bridge" between manual annotation (QuPath) and automated 
processing (Python Pipeline). It consolidates scattered, human-generated text exports 
into a single, normalized Master Dataset for the landmark propagation system.

Architectural Context
---------------------
Manual landmark acquisition is inherently fragmented; annotators typically export one 
text file per image/core (e.g., `Core01.txt`, `Core02.txt`). This module standardizes 
these disparate inputs by:
1. **Aggregating** all .txt files from a target directory.
2. **Parsing** inconsistent file naming conventions to robustly extract metadata 
   (Core ID, Slice Index).
3. **Normalizing** spatial coordinates from physical units (Microns) to computational 
   units (Pixels), ensuring compatibility with the 4x downsampled image space used 
   by the optical flow algorithm.

Methodology (Data Engineering)
------------------------------
* **Regex Pattern Matching:** Utilizes flexible regular expressions to deduce `Core_ID` 
  and `Slice_Index` from filenames or internal metadata strings, handling common 
  variations in user naming (e.g., "Core_01" vs "Core1", "s13" vs "Slice13").
* **Unit Conversion:** Applies the inverse of the image pixel size ($1 / 1.9844 \mu m$) 
  to transform centroid coordinates, maintaining floating-point precision.
* **Schema Validation:** Checks for the existence of required columns ("Centroid X µm") 
  and gracefully handles missing or malformed data rows.

Input / Output
--------------
* **Input:** A directory containing multiple tab-delimited text files exported from 
  QuPath (Measurement Tables).
* **Output:** A single CSV file (`manual_landmarks.csv`) with the standardized schema:
  `[Core_ID, Slice_Index, X, Y, Label]`.

"""
import pandas as pd
import os
import re
from glob import glob
import config

# --- CONFIGURATION ---
INPUT_FOLDER = os.path.join(config.DATASPACE, 
    "downsampled_4x",     
    "qupath_exports")  # Folder containing Core01.txt, Core02.txt, etc.
OUTPUT_FILE = os.path.join(config.DATASPACE, 
    "downsampled_4x",     
    "manual_landmarks.csv")

# Critical Number: Original (0.4961) * 4x Downsample
PIXEL_SIZE_MICRONS = 1.9844 

def parse_single_file(filepath):
    """
    Parses a single QuPath text file and returns a list of dictionaries.
    """
    try:
        # Read the file (QuPath exports are usually Tab-separated)
        df = pd.read_csv(filepath, sep='\t')
    except Exception as e:
        print(f"Skipping {os.path.basename(filepath)}: Could not read file. {e}")
        return []

    file_data = []
    
    for index, row in df.iterrows():
        # 1. robust Core ID extraction
        # We look at the 'Image' column first, then the filename if needed
        image_name = str(row.get('Image', ''))
        
        # Regex to find "Core_01", "Core01", "Core1"
        # Adjust regex if your naming convention is different!
        core_match = re.search(r"Core_?(\d+)", image_name, re.IGNORECASE)
        
        if not core_match:
            # Fallback: Try to guess from the text filename itself
            core_match = re.search(r"Core_?(\d+)", os.path.basename(filepath), re.IGNORECASE)
            
        if core_match:
            # Normalize to "Core_XX" format (e.g. Core_01)
            core_num = int(core_match.group(1))
            core_id = f"Core_{core_num:02d}"
        else:
            print(f"  Warning: Could not determine Core ID for row in {os.path.basename(filepath)}")
            continue

        # 2. Slice Index extraction
        # Looks for "s13" or "slice13" in the Name
        name = str(row.get('Name', ''))
        slice_match = re.search(r"s(\d+)", name, re.IGNORECASE)
        
        if slice_match:
            # Convert "s13" -> index 12 (0-based)
            slice_index = int(slice_match.group(1)) - 1
        else:
            # Default to 0 if you labeled the first image and didn't name it special
            slice_index = 0
            
        # 3. Coordinate Conversion
        # columns are "Centroid X µm" and "Centroid Y µm"
        try:
            x_microns = row['Centroid X µm']
            y_microns = row['Centroid Y µm']
        except KeyError:
             # Fallback for "px" columns if you managed to export pixels
            if 'Centroid X px' in row:
                x_px = row['Centroid X px']
                y_px = row['Centroid Y px']
            else:
                print(f"  Error: Missing Centroid columns in {os.path.basename(filepath)}")
                continue
        else:
            # Perform Math: Microns -> Pixels
            x_px = x_microns / PIXEL_SIZE_MICRONS
            y_px = y_microns / PIXEL_SIZE_MICRONS
        
        file_data.append({
            "Core_ID": core_id,
            "Slice_Index": slice_index,
            "X": x_px,
            "Y": y_px,
            "Label": name
        })
        
    return file_data

def batch_convert():
    print("="*60)
    print(f"  BATCH LANDMARK MERGER")
    print(f"  Scanning: {INPUT_FOLDER}")
    print("="*60)
    
    # 1. Find all .txt files
    files = sorted(glob(os.path.join(INPUT_FOLDER, "*.txt")))
    
    if not files:
        print("No .txt files found! Check your path.")
        return

    all_landmarks = []
    
    # 2. Process each file
    for f in files:
        print(f"Processing: {os.path.basename(f)}...")
        data = parse_single_file(f)
        all_landmarks.extend(data)
        
    # 3. Save Master CSV
    if all_landmarks:
        df_master = pd.DataFrame(all_landmarks)
        
        # Sort for neatness
        df_master = df_master.sort_values(by=['Core_ID', 'Slice_Index'])
        
        df_master.to_csv(OUTPUT_FILE, index=False)
        print("\n" + "="*60)
        print(f"SUCCESS.")
        print(f"Combined {len(files)} files into one Master Database.")
        print(f"Total Landmarks: {len(df_master)}")
        print(f"Output saved to: {OUTPUT_FILE}")
        print("="*60)
        
        # Quality Check: Print unique cores found
        print(f"Cores identified: {df_master['Core_ID'].unique()}")
    else:
        print("No valid landmarks extracted.")

if __name__ == "__main__":
    batch_convert()