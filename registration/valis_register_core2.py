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
import re
import logging
import yaml

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


def load_slice_filter(yaml_path, core_name):
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh) or {}
    raw = data.get(core_name)
    if raw is None:
        return None
    allowed = set()
    for part in str(raw).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            allowed.update(range(int(lo.strip()), int(hi.strip()) + 1))
        else:
            allowed.add(int(part))
    return allowed


def get_slice_number(filename):
    match = re.search(r"TMA_(\d+)_", os.path.basename(filename))
    if not match:
        raise ValueError(
            f"Could not parse slice number from filename: {os.path.basename(filename)} "
            f"(expected pattern 'TMA_<digits>_'). Fix the regex or the filename convention "
            f"before continuing — silently defaulting to slice 0 will corrupt slice filtering."
        )
    return int(match.group(1))


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
    DATA_BASE_PATH    = os.path.join(config.DATASPACE, "TMA_Cores_Grouped_Rotate_Conformed")
    input_dir         = os.path.join(DATA_BASE_PATH, args.core_name)
    WORK_OUTPUT       = os.path.join(config.DATASPACE, "VALIS_Filter_Eval")
    output_dir        = os.path.join(WORK_OUTPUT, args.core_name)
    reg_slides_dir    = os.path.join(output_dir, "registered_slides")
    SLICE_FILTER_YAML = os.path.join(config.DATASPACE, "slice_filter.yaml")

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
        glob.glob(os.path.join(input_dir, "*.tiff")),
        key=get_slice_number
    )
    if not sample_files:
        logger.error("No TIFF files found.")
        sys.exit(1)

    valid_files = [f for f in sample_files if "_thumb" not in os.path.basename(f)]

    allowed_positions = load_slice_filter(SLICE_FILTER_YAML, args.core_name)
    if allowed_positions is not None:
        original_count = len(valid_files)
        kept_ids       = [get_slice_number(f) for i, f in enumerate(valid_files) if i in allowed_positions]
        valid_files    = [f for i, f in enumerate(valid_files) if i in allowed_positions]
        excluded       = original_count - len(valid_files)
        logger.info(
            f"Slice filter active: keeping {len(valid_files)}/{original_count} slices "
            f"(0-indexed positions {sorted(allowed_positions)} -> TMA IDs {kept_ids}), "
            f"{excluded} excluded."
        )
        if len(valid_files) == 0:
            logger.error("Slice filter excluded all slices.")
            sys.exit(1)
    else:
        logger.info(f"No slice filter — using all {len(valid_files)} slices.")
    
    # Staging directory logic preserved to protect I/O 
    staging_dir = os.path.join(WORK_OUTPUT, args.core_name, "staging")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)
    
    # valid_files is already sorted numerically by get_slice_number() (see sample_files
    # above). imgs_ordered=True makes VALIS trust the *alphabetical* order it sees in
    # src_dir, so we zero-pad the symlink names here to force alphabetical order to
    # match numeric slice order. The original files on disk are untouched — only the
    # symlink names in staging_dir are padded.
    pad_width = max(3, len(str(get_slice_number(valid_files[-1]))))
    for file_path in valid_files:
        slice_num = get_slice_number(file_path)
        padded_name = f"{slice_num:0{pad_width}d}_{os.path.basename(file_path)}"
        target_link = os.path.join(staging_dir, padded_name)
        os.symlink(file_path, target_link)

    logger.info(f"Staged {len(valid_files)} files in {staging_dir} (zero-padded for ordering)")

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
            name=args.core_name,
            imgs_ordered=True
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