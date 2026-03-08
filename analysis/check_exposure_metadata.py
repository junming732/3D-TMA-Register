"""
check_exposure_metadata.py

Reads metadata from OME-TIFF and PerkinElmer/Akoya .qptiff files and prints
per-channel information including exposure times, unmixing status, and channel names.

Handles:
  - Standard OME-XML (tifffile.is_ome)
  - PerkinElmer-QPI format (PhenoImager / Vectra) — parses per-page XML
    to reconstruct the full channel table including IsUnmixedComponent status

Usage:
    python check_exposure_metadata.py --file /path/to/scan.qptiff
    python check_exposure_metadata.py --input_folder /path/to/TMA_Core_Folder
    python check_exposure_metadata.py --input_folder /path/to/folder --n_slices 3
    python check_exposure_metadata.py --file /path/to/scan.qptiff --dump_full_xml
"""

import os
import sys
import argparse
import glob
import re
import xml.etree.ElementTree as ET

import tifffile

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Check TIFF channel / exposure metadata.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--input_folder", type=str, help="Folder containing .ome.tif/.qptiff slices")
group.add_argument("--file",         type=str, help="Single .ome.tif or .qptiff file")
parser.add_argument("--n_slices",    type=int, default=1,
                    help="How many slices to inspect from --input_folder (default: 1)")
parser.add_argument("--ck_channel",  type=int, default=6,
                    help="CK channel index (default: 6)")
parser.add_argument("--dump_full_xml", action="store_true",
                    help="Dump full per-page XML to <filename>_page_xml/ folder")
args = parser.parse_args()

CK_IDX = args.ck_channel


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_files(folder, n):
    patterns = ["*.ome.tif", "*.qptiff", "*.tif", "*.tiff"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat)))
    files = sorted(set(files),
                   key=lambda f: int(m.group(1))
                   if (m := re.search(r"TMA_(\d+)_", os.path.basename(f))) else 0)
    if not files:
        print(f"[ERROR] No TIFF files found in {folder}")
        sys.exit(1)
    return files[:n]


# ─────────────────────────────────────────────────────────────────────────────
# PerkinElmer-QPI parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_perkinelmer_qpi(tif, dump_xml=False, filepath=""):
    """
    PerkinElmer qptiff stores one XML ImageDescription per page (one page = one
    channel at one resolution level). We read only the FullResolution pages and
    reconstruct the channel table.
    Returns list of dicts, one per channel.
    """
    channels  = []
    page_xmls = []

    for page_idx, page in enumerate(tif.pages):
        desc_tag = page.tags.get(270)
        if desc_tag is None:
            continue
        raw = desc_tag.value
        if isinstance(raw, bytes):
            # PhenoImager writes utf-16 with BOM
            for enc in ("utf-16", "utf-8"):
                try:
                    raw = raw.decode(enc, errors="replace")
                    break
                except Exception:
                    continue

        if "<PerkinElmer-QPI-ImageDescription>" not in raw:
            continue

        # Strip BOM if present before parsing
        xml_str = raw.lstrip("\ufeff")
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            print(f"  [!] XML parse error on page {page_idx}: {e}")
            continue

        def g(tag, default="—"):
            el = root.find(tag)
            return el.text.strip() if el is not None and el.text else default

        image_type = g("ImageType")
        # Skip sub-resolution pyramid pages
        if image_type not in ("FullResolution", "—"):
            continue

        ch = {
            "page_index":          page_idx,
            "ImageType":           image_type,
            "Name":                g("Name"),
            "ExposureTime":        g("ExposureTime"),
            "ExposureTimeUnit":    g("ExposureTimeUnit", "ms"),
            "IsUnmixedComponent":  g("IsUnmixedComponent"),
            "Fluor":               g("Fluor"),
            "Color":               g("Color"),
            "AcquisitionSoftware": g("AcquisitionSoftware"),
            "SignalUnits":         g("SignalUnits"),
        }

        page_xmls.append((page_idx, xml_str))
        channels.append(ch)

    for i, ch in enumerate(channels):
        ch["channel_index"] = i

    if dump_xml and page_xmls:
        xml_dir = os.path.splitext(filepath)[0] + "_page_xml"
        os.makedirs(xml_dir, exist_ok=True)
        for pidx, xml_str in page_xmls:
            with open(os.path.join(xml_dir, f"page_{pidx:04d}.xml"), "w") as f:
                f.write(xml_str)
        print(f"\n  Per-page XML written to: {xml_dir}/")

    return channels


# ─────────────────────────────────────────────────────────────────────────────
# Standard OME-XML parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_ome_xml(xml_str):
    root   = ET.fromstring(xml_str)
    tag    = root.tag
    ns     = tag[1:tag.index("}")] if tag.startswith("{") else ""
    prefix = f"{{{ns}}}" if ns else ""

    channels = []
    for pixels in root.iter(f"{prefix}Pixels"):
        for idx, ch in enumerate(pixels.findall(f"{prefix}Channel")):
            info = {"channel_index": idx}
            for attr in ["Name", "ExposureTime", "ExposureTimeUnit",
                         "Fluor", "EmissionWavelength", "ExcitationWavelength", "Color"]:
                val = ch.get(attr)
                if val is not None:
                    info[attr] = val
            channels.append(info)
        break
    return channels


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def summarise_channels(channels, ck_idx, filepath):
    n_ch   = len(channels)
    af_idx = n_ch - 1

    print(f"\n{'='*80}")
    print(f"File    : {os.path.basename(filepath)}")
    print(f"Channels: {n_ch}")
    print(f"{'='*80}")

    if not channels:
        print("  [!] No channel metadata could be parsed.")
        return

    print(f"\n{'Idx':<5} {'Name':<28} {'ExposureTime':<14} {'Unit':<6} "
          f"{'IsUnmixed':<12} {'Fluor'}")
    print("-" * 85)

    for ch in channels:
        idx     = ch.get("channel_index", "?")
        name    = ch.get("Name", "—")[:27]
        exp     = ch.get("ExposureTime", "—")
        unit    = ch.get("ExposureTimeUnit", "—")
        unmixed = ch.get("IsUnmixedComponent", "—")
        fluor   = ch.get("Fluor", "—")

        marker = ""
        if idx == ck_idx:  marker = "  ← CK"
        if idx == af_idx:  marker += "  ← last (AF?)"

        print(f"  {idx:<4} {name:<28} {exp:<14} {unit:<6} {unmixed:<12} {fluor}{marker}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("─" * 80)
    print("SUMMARY")
    print("─" * 80)

    unmixed_chs = [c for c in channels
                   if c.get("IsUnmixedComponent", "").strip().lower() == "true"]

    if unmixed_chs:
        print(f"\n  ⚠  IsUnmixedComponent=True on {len(unmixed_chs)} channel(s).")
        print("     This file was spectrally UNMIXED by PhenoImager before you received it.")
        print("     The last channel is the UNMIXED AUTOFLUORESCENCE COMPONENT —")
        print("     it is NOT a raw blank acquisition.")
        print()
        print("     What this means for subtraction:")
        print("       - Spectral unmixing already separated AF from all other channels.")
        print("       - The CK channel you see in QuPath is already AF-corrected.")
        print("       - Subtracting the AF component again would DOUBLE-correct and")
        print("         likely destroy CK signal — this is probably why you see fewer")
        print("         keypoints after subtraction.")
        print()
        print("     RECOMMENDATION: Do NOT subtract the AF channel in your pipeline.")
        print("     Use CK channel (idx {}) as-is for feature detection.".format(ck_idx))
    else:
        print("\n  IsUnmixedComponent not found — file appears to be raw (not unmixed).")

        ck_ch = next((c for c in channels if c.get("channel_index") == ck_idx), None)
        af_ch = next((c for c in channels if c.get("channel_index") == af_idx), None)

        if ck_ch and af_ch:
            ck_exp = ck_ch.get("ExposureTime", "—")
            af_exp = af_ch.get("ExposureTime", "—")
            print(f"\n  CK  (idx {ck_idx}) '{ck_ch.get('Name','?')}': "
                  f"ExposureTime = {ck_exp} {ck_ch.get('ExposureTimeUnit','')}")
            print(f"  AF  (idx {af_idx}) '{af_ch.get('Name','?')}': "
                  f"ExposureTime = {af_exp} {af_ch.get('ExposureTimeUnit','')}")
            try:
                ratio = float(ck_exp) / float(af_exp)
                print(f"\n  Exposure ratio CK/AF = {ratio:.4f}")
                if abs(ratio - 1.0) < 0.01:
                    print("  ✓ Equal exposures — plain subtraction (CK - AF) is correct.")
                else:
                    print(f"  ✗ Exposures differ — use scaled subtraction:")
                    print(f"      corrected_CK = CK - AF * {ratio:.4f}")
            except (ValueError, ZeroDivisionError):
                print("  [!] Could not compute exposure ratio.")


# ─────────────────────────────────────────────────────────────────────────────
# Per-file inspector
# ─────────────────────────────────────────────────────────────────────────────

def inspect_file(filepath):
    with tifffile.TiffFile(filepath) as tif:
        channels = []

        # 1. Standard OME-XML
        if tif.is_ome:
            try:
                channels = parse_ome_xml(tif.ome_metadata)
            except Exception as e:
                print(f"  [!] OME-XML parse error: {e}")

        # 2. PerkinElmer-QPI
        if not channels:
            try:
                channels = parse_perkinelmer_qpi(
                    tif, dump_xml=args.dump_full_xml, filepath=filepath)
            except Exception as e:
                print(f"  [!] PerkinElmer-QPI parse error: {e}")

        # 3. Fallback
        if not channels:
            print(f"\n{'='*80}")
            print(f"File: {os.path.basename(filepath)}")
            print("  [!] Could not parse channel metadata. Raw ImageDescription (page 0):")
            desc_tag = tif.pages[0].tags.get(270)
            if desc_tag:
                raw = desc_tag.value
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                print(raw[:2000])
            return

        summarise_channels(channels, CK_IDX, filepath)

        try:
            shape = tif.series[0].shape
            print(f"\n  Series shape : {shape}  dtype: {tif.series[0].dtype}")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    files = [args.file] if args.file else find_files(args.input_folder, args.n_slices)
    for f in files:
        inspect_file(f)
    print()


if __name__ == "__main__":
    main()