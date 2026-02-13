import os
import sys
import config

# --- IMPORT YOUR EXISTING MODULES ---
# (Ensure these .py files are in the same directory)
try:
    import batch_convert_landmarks as step1
    import propagate_landmarks as step2
    import visualize_landmarks as step3
except ImportError as e:
    print("ERROR: Could not find one of the pipeline scripts.")
    print(f"Missing: {e.name}.py")
    print("Please ensure all 3 scripts are saved in this folder.")
    sys.exit(1)

# --- CONFIGURATION FOR THE TEST ---
# We override the 'Global Variables' in the modules dynamically.
# This keeps your original scripts clean!

# 1. SETUP PATHS
# Where your 7 text files are:
TEST_LABELS_DIR = INPUT_FOLDER = os.path.join(config.DATASPACE, 
    "downsampled_4x",     
    "qupath_exports") 

# Where your 4x images are (The Big Folder):
IMAGES_ROOT_DIR =INPUT_FOLDER = os.path.join(config.DATASPACE, 
    "downsampled_4x")

# Where we will save the intermediate CSV/JSON:
TEST_OUTPUT_DIR = INPUT_FOLDER = os.path.join(config.DATASPACE, 
    "downsampled_4x",     
    "test_results")

# --- EXECUTION ENGINE ---

def run_test_pipeline():
    print("="*60)
    print("  STARTING 7-CORE PILOT TEST")
    print("="*60)

    if not os.path.exists(TEST_LABELS_DIR):
        print(f"Error: Create a folder '{TEST_LABELS_DIR}' and put your 7 text files there first.")
        return

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: BATCH CONVERT (Text Files -> CSV)
    # ---------------------------------------------------------
    print("\n--- STEP 1: CONVERTING LABELS ---")
    
    # Override Step 1 Config
    step1.INPUT_FOLDER = TEST_LABELS_DIR
    step1.OUTPUT_FILE = os.path.join(TEST_OUTPUT_DIR, "manual_landmarks.csv")
    
    # Run Step 1
    step1.batch_convert()

    # ---------------------------------------------------------
    # STEP 2: PROPAGATE (CSV + Images -> JSON)
    # ---------------------------------------------------------
    print("\n--- STEP 2: RUNNING OPTICAL FLOW ---")
    
    # Override Step 2 Config
    step2.INPUT_ROOT = IMAGES_ROOT_DIR
    step2.LANDMARK_FILE = step1.OUTPUT_FILE  # Point to the CSV we just made
    step2.OUTPUT_JSON = os.path.join(TEST_OUTPUT_DIR, "propagated_landmarks.json")
    
    # Run Step 2
    step2.run_propagation()

    # ---------------------------------------------------------
    # STEP 3: VISUALIZE (JSON + Images -> GIFs)
    # ---------------------------------------------------------
    print("\n--- STEP 3: GENERATING QC GIFS ---")
    
    # Override Step 3 Config
    step3.INPUT_ROOT = IMAGES_ROOT_DIR
    step3.JSON_FILE = step2.OUTPUT_JSON      # Point to the JSON we just made
    step3.OUTPUT_QC_DIR = os.path.join(TEST_OUTPUT_DIR, "QC_GIFs")
    
    # Run Step 3
    step3.run_visualization()

    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print(f"  Check your GIFs here: {step3.OUTPUT_QC_DIR}")
    print("="*60)

if __name__ == "__main__":
    run_test_pipeline()