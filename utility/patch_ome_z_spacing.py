"""Updates OME-XML attributes in an existing OME-TIFF file in place.

This script modifies the ImageDescription tag by reading and writing raw 
UTF-8 bytes directly at the file offset. This approach avoids expensive 
pixel data rewrites and circumvents the 7-bit ASCII limitation of 
tifffile.tiffcomment(), which fails on standard OME-XML characters 
(e.g., PhysicalSizeZUnit="µm").

Notes:
    * In-place modification requires the new attribute string to fit exactly 
      within the existing byte length of the tag. The script automatically 
      zero-pads decimals to ensure byte-length parity.
    * To guarantee file stability and prevent partial writes, an exception 
      is raised before modification if the new value exceeds the original 
      byte capacity.
    * A .bak copy of the target file is generated automatically before execution 
      unless explicitly disabled via --no_backup.

Examples:
    Update a single file to a new Z-spacing of 4.0:
        $ python patch_ome_z_spacing.py --path core_Aligned.ome.tif --new_z 4.0
        
    Run a dry-run test across multiple files using a glob pattern:
        $ python patch_ome_z_spacing.py --glob "OUTPUT_FOLDER/*/*_Aligned.ome.tif" --new_z 4.0 --dry_run
"""

import argparse
import glob
import re
import shutil
import sys

import tifffile


def build_pattern(attr):
    return re.compile(rf'{attr}="([0-9.]+)"')


def patch_file(path, attr, new_value, dry_run=False, backup=True):
    with tifffile.TiffFile(path) as tf:
        tag = tf.pages[0].tags.get("ImageDescription")
        if tag is None:
            print(f"[SKIP] {path}: no ImageDescription tag found.")
            return False
        valueoffset = tag.valueoffset
        count = tag.count

    with open(path, "rb") as f:
        f.seek(valueoffset)
        raw = f.read(count)

    xml = raw.decode("utf-8")

    pattern = build_pattern(attr)
    match = pattern.search(xml)
    if not match:
        print(f"[SKIP] {path}: no {attr} attribute found in OME-XML.")
        return False

    old_str = match.group(0)        # e.g. PhysicalSizeZ="4.5"
    old_val_str = match.group(1)    # e.g. "4.5"

    new_val_str = f"{new_value:g}"
    if len(new_val_str) < len(old_val_str):
        if "." not in new_val_str:
            new_val_str += "."
        new_val_str = new_val_str.ljust(len(old_val_str), "0")
    elif len(new_val_str) > len(old_val_str):
        raise ValueError(
            f"{path}: new value '{new_val_str}' needs more characters than "
            f"old value '{old_val_str}' — can't grow the tag in place. "
            f"Use more decimal places so the padded string is "
            f"<= {len(old_val_str)} chars (e.g. pass 4.0 not 4)."
        )

    new_str = f'{attr}="{new_val_str}"'
    new_xml = xml.replace(old_str, new_str)
    new_raw = new_xml.encode("utf-8")

    if len(new_raw) != len(raw):
        raise ValueError(
            f"{path}: encoded byte length changed ({len(raw)} -> "
            f"{len(new_raw)}), refusing to write."
        )

    print(f"[{'DRY RUN' if dry_run else 'PATCH'}] {path}")
    print(f"    {old_str}  ->  {new_str}   ({len(raw)} bytes, offset {valueoffset})")

    if dry_run:
        return True

    if backup:
        bak_path = path + ".bak"
        shutil.copy2(path, bak_path)
        print(f"    backup written: {bak_path}")

    with open(path, "r+b") as f:
        f.seek(valueoffset)
        f.write(new_raw)

    check = tifffile.tiffcomment(path)
    if new_str not in check:
        print("    [WARNING] verification failed — new value not found after write.")
        return False
    print("    verified OK.")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", type=str, help="Single OME-TIFF file to patch.")
    ap.add_argument("--glob", type=str, help="Glob pattern matching multiple OME-TIFF files.")
    ap.add_argument("--attr", type=str, default="PhysicalSizeZ",
                     help="OME-XML attribute to patch (default: PhysicalSizeZ).")
    ap.add_argument("--new_z", type=float, required=True,
                     help="Correct value to write, e.g. 4.0")
    ap.add_argument("--dry_run", action="store_true",
                     help="Show what would change without writing anything.")
    ap.add_argument("--no_backup", action="store_true",
                     help="Skip writing a .bak copy before patching.")
    args = ap.parse_args()

    if not args.path and not args.glob:
        ap.error("Provide either --path or --glob.")

    files = [args.path] if args.path else sorted(glob.glob(args.glob))
    if not files:
        print("No files matched.")
        sys.exit(1)

    n_ok = 0
    for f in files:
        try:
            if patch_file(f, args.attr, args.new_z, dry_run=args.dry_run,
                           backup=not args.no_backup):
                n_ok += 1
        except Exception as exc:
            print(f"[ERROR] {f}: {exc}")

    verb = "would be patched" if args.dry_run else "patched"
    print(f"\n{n_ok}/{len(files)} file(s) {verb}.")


if __name__ == "__main__":
    main()