"""Verifies the metadata and pixel integrity of patched OME-TIFF files.

This script validates the output of `patch_ome_z_spacing.py` across a batch 
of files to ensure in-place modifications succeeded without corrupting the 
underlying image data or XML structure.

Notes:
    * Validation ensures the `PhysicalSizeZ` attribute matches the expected 
      target value across all evaluated files.
    * Confirms the OME-XML remains well-formed and strictly parsable after 
      the byte-level string replacement.
    * Verifies that the pixel data is byte-identical to the original `.bak` 
      backup, guaranteeing that the registered volume was untouched.

Example:
    Run verification across a batch of files for an expected Z-spacing of 4.0:
        $ python verify_z_patch.py --glob "/path/to/WORK_OUTPUT/*/*_Aligned.ome.tif" --expected_z 4.0
"""

import argparse
import glob
import sys
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


def verify_file(path, expected_z, check_pixels=True):
    ok = True
    xml = tifffile.tiffcomment(path)

    # 1. XML well-formed
    try:
        ET.fromstring(xml)
    except ET.ParseError as exc:
        print(f"[FAIL] {path}: XML no longer parses — {exc}")
        return False

    # 2. Correct value present
    tag = f'PhysicalSizeZ="{expected_z:g}'
    if tag not in xml and f'PhysicalSizeZ="{expected_z}"' not in xml:
        # be lenient about trailing-zero formatting, extract and compare numerically
        import re
        m = re.search(r'PhysicalSizeZ="([0-9.]+)"', xml)
        if not m or abs(float(m.group(1)) - expected_z) > 1e-6:
            found = m.group(1) if m else "NOT FOUND"
            print(f"[FAIL] {path}: PhysicalSizeZ={found}, expected {expected_z}")
            ok = False

    # 3. Pixel data unchanged vs backup, if a .bak exists
    if check_pixels:
        bak = path + ".bak"
        try:
            a = tifffile.imread(bak)
            b = tifffile.imread(path)
            if not np.array_equal(a, b):
                print(f"[FAIL] {path}: pixel data differs from {bak}!")
                ok = False
        except FileNotFoundError:
            print(f"[WARN] {path}: no .bak found, skipping pixel-integrity check.")

    if ok:
        print(f"[OK]   {path}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, required=True)
    ap.add_argument("--expected_z", type=float, required=True)
    ap.add_argument("--no_pixel_check", action="store_true",
                     help="Skip comparing against .bak (faster, less thorough).")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print("No files matched.")
        sys.exit(1)

    results = [verify_file(f, args.expected_z, check_pixels=not args.no_pixel_check)
               for f in files]

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(files)} file(s) verified OK.")
    if n_ok != len(files):
        sys.exit(1)


if __name__ == "__main__":
    main()