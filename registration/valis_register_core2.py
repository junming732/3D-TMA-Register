"""
VALIS Registration Pipeline — Baseline Automated Method
======================================================================================
This script integrates the baseline VALIS automated pipeline into the existing 
I/O and staging architecture. 

It removes manual multi-channel processing and non-rigid grid constraints, 
relying instead on the built-in heuristics described in the VALIS documentation 
for modality detection, feature extraction, and deformation scaling.
"""

import os
import sys
import time
import argparse
import shutil
import glob
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

try:
    from valis import registration
except ImportError as e:
    logger.critical(f"Could not import valis: {e}")
    sys.exit(1)

try:
    import config
except ImportError:
    logger.critical("Could not import 'config.py'.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline VALIS registration for TMA cores."
    )
    parser.add_argument("--core_name", type=str, required=True,
                        help="Name of the core folder.")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. Path Definitions & I/O Setup
    # -------------------------------------------------------------------------
    # Restored exact path routing from the original script
    DATA_BASE_PATH = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate")
    input_dir      = os.path.join(DATA_BASE_PATH, args.core_name)
    WORK_OUTPUT    = os.path.join(config.DATASPACE, "VALIS_Baseline_Eval")
    output_dir     = os.path.join(WORK_OUTPUT, args.core_name)
    reg_slides_dir = os.path.join(output_dir, "registered_slides")

    logger.info("=" * 60)
    logger.info(f"Baseline VALIS Registration | Core: {args.core_name}")
    logger.info("=" * 60)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. File Discovery & Staging
    # -------------------------------------------------------------------------
    sample_files = sorted(
        glob.glob(os.path.join(input_dir, "*.tif")) +
        glob.glob(os.path.join(input_dir, "*.tiff"))
    )
    if not sample_files:
        logger.error("No TIFF files found.")
        sys.exit(1)

    valid_files = [f for f in sample_files if "_thumb" not in os.path.basename(f)]
    
    # Staging directory logic preserved to protect I/O 
    staging_dir = os.path.join(WORK_OUTPUT, args.core_name, "staging")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)
    
    for file_path in valid_files:
        target_link = os.path.join(staging_dir, os.path.basename(file_path))
        os.symlink(file_path, target_link)
        
    logger.info(f"Staged {len(valid_files)} files in {staging_dir}")

    # -------------------------------------------------------------------------
    # 3. Automated VALIS Pipeline
    # -------------------------------------------------------------------------
    try:
        t0 = time.time()
        logger.info("Step 1/3 — Automated Registration...")
        
        # Baseline Instantiation: No custom processors or rigid fallback definitions.
        # VALIS will automatically determine modality and handle non-rigid scaling.
        registrar = registration.Valis(
            src_dir=staging_dir,
            dst_dir=output_dir,
            name=args.core_name
        )
        
        registrar.register()
        logger.info(f"  Registration completed in {time.time()-t0:.1f}s")

        t1 = time.time()
        logger.info("Step 2/3 — Warping Registered Slides...")
        
        # crop="overlap" ensures clean boundaries for the final stack
        registrar.warp_and_save_slides(
            dst_dir=reg_slides_dir,
            crop="overlap"
        )
        logger.info(f"  Warping completed in {time.time()-t1:.1f}s")

        t2 = time.time()
        logger.info("Step 3/3 — Merging into OME-TIFF stack...")
        
        merged_path = os.path.join(
            output_dir, f"{args.core_name}_VALIS_baseline.ome.tif"
        )
        
        # Utilizing the built-in VALIS merge method as defined in the official documentation
        # instead of a custom stacking function.
        registrar.warp_and_merge_slides(
            merged_path,
            drop_duplicates=True
        )
        logger.info(f"  Merging completed in {time.time()-t2:.1f}s")

        logger.info("=" * 60)
        logger.info(f"Registration complete in {time.time()-t0:.1f}s total.")
        logger.info(f"Final stack: {merged_path}")

    except Exception as exc:
        logger.error(f"Registration failed: {exc}")
        raise

    finally:
        # Crucial for stable pipeline execution across multiple runs
        registration.kill_jvm()

if __name__ == "__main__":
    main()