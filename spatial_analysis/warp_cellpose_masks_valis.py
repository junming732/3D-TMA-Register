"""
warp_cellpose_masks_valis.py
=============================
VALIS-variant counterpart to warp_cellpose_masks.py.

CLI contract is identical to warp_cellpose_masks.py on purpose, so the
Snakefile's `warp_cellpose_masks` rule can swap scripts per-variant
without changing the rule's input/output/param shape:

    python warp_cellpose_masks_valis.py \
        --core_name Core_01 \
        --mask_dir   <DATASPACE>/<cellpose.mask_dir_name>/Core_01 \
        --deform_dir <DATASPACE>/<registration.valis.output_dir_name>/Core_01/data \
        --out_dir    <DATASPACE>/<cellpose.warped_dir_name>/Core_01 \
        --plot_qc

--deform_dir here is NOT the deformation_maps/ folder — for the valis
variant it should point at the "data" directory VALIS writes under its
own dst_dir (or directly at the *_registrar.pickle file; both are
accepted, see resolve_registrar_path()). Wire this via each registration
block's `deform_subdir_name` in config.yaml (see valis_register_core2.py
notes) rather than hardcoding "deformation_maps" for every variant.

Usage
-----
    python warp_cellpose_masks_valis.py --core_name Core_01 \
        --mask_dir /path/to/masks/Core_01 \
        --deform_dir /path/to/VALIS_out/Core_01/data \
        --out_dir /path/to/warped/Core_01 --plot_qc

    # Debug what your VALIS version's Slide object actually exposes:
    python warp_cellpose_masks_valis.py --core_name Core_01 \
        --mask_dir ... --deform_dir ... --out_dir ... --inspect_only
"""

import os
import re
import sys
import glob
import logging
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import tifffile

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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Warp CellPose masks into VALIS-registered space using a "
                "saved Valis registrar (instead of a deformation_maps/*.npz)."
)
parser.add_argument("--core_name", type=str, required=True)
parser.add_argument("--mask_dir", type=str, required=True,
                     help="Directory of per-slice raw CellPose masks "
                          "(same directory cellpose_segmentation.py wrote).")
parser.add_argument("--deform_dir", type=str, required=True,
                     help="Directory containing the saved Valis registrar "
                          "(VALIS's own '<dst_dir>/data' folder), or a "
                          "direct path to a '*_registrar.pickle' file.")
parser.add_argument("--out_dir", type=str, required=True,
                     help="Directory to write warped masks into.")
parser.add_argument("--plot_qc", action="store_true",
                     help="Save a raw-vs-warped overlay PNG per slice under "
                          "<out_dir>/qc/.")
parser.add_argument("--workers", type=int, default=4,
                     help="Thread-pool size for per-slice warping (default 4). "
                          "Threads, not processes: VALIS/JVM objects are not "
                          "guaranteed pickle-safe across process boundaries.")
parser.add_argument("--interp", type=str, default="nearest",
                     choices=["nearest"],
                     help="Interpolation for warping label masks. Always "
                          "nearest-neighbor — anything else corrupts label IDs.")
parser.add_argument("--inspect_only", action="store_true",
                     help="Load the registrar, print the first slide object's "
                          "available attributes/methods, and exit without "
                          "warping anything. Use this to confirm the warp "
                          "API name/signature for your installed VALIS "
                          "version before trusting this script.")
args = parser.parse_args()

TARGET_CORE = args.core_name

# Same convention as valis_register_core2.get_slice_number() — deliberately
# duplicated rather than imported, since each pipeline script in this repo
# is meant to run standalone via the Snakefile.
_SLICE_RE = re.compile(r"TMA_(\d+)_")


def get_slice_number(name: str) -> int:
    match = _SLICE_RE.search(os.path.basename(str(name)))
    if not match:
        raise ValueError(
            f"Could not parse slice number from: {name} "
            f"(expected 'TMA_<digits>_' substring). Fix the regex or the "
            f"naming convention before continuing — silently defaulting "
            f"would mismatch a mask against the wrong slide's transform."
        )
    return int(match.group(1))


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRAR LOADING
# ─────────────────────────────────────────────────────────────────────────────
def resolve_registrar_path(deform_dir: str) -> str:
    """Accept either a direct path to a *_registrar.pickle, or a directory
    (VALIS's own 'data' folder) containing exactly one."""
    if os.path.isfile(deform_dir):
        return deform_dir
    if not os.path.isdir(deform_dir):
        raise FileNotFoundError(f"--deform_dir does not exist: {deform_dir}")
    candidates = sorted(glob.glob(os.path.join(deform_dir, "*_registrar.pickle")))
    if not candidates:
        raise FileNotFoundError(
            f"No '*_registrar.pickle' found under {deform_dir}. "
            f"Confirm valis_register_core2.py is actually saving the "
            f"registrar (e.g. registrar.save(...)) into this folder — "
            f"as of the current version it is NOT, that needs adding."
        )
    if len(candidates) > 1:
        logger.warning(
            f"Multiple registrar pickles found under {deform_dir}; "
            f"using the most recent: {candidates[-1]}"
        )
    return candidates[-1]


def load_registrar(deform_dir: str):
    pickle_path = resolve_registrar_path(deform_dir)
    logger.info(f"Loading saved Valis registrar: {pickle_path}")
    # NOTE: confirm this matches your installed valis-wsi version's public
    # API (see module docstring, point 1).
    registrar = registration.load_registrar(pickle_path)
    return registrar


def build_slice_to_slide_map(registrar) -> dict:
    """slice_number (int) -> VALIS Slide object, recovered from whichever
    filename-bearing attribute the installed VALIS version exposes."""
    slide_dict = getattr(registrar, "slide_dict", None)
    if not slide_dict:
        raise AttributeError(
            "Loaded registrar has no 'slide_dict' attribute — VALIS's "
            "internal API has likely changed. Run with --inspect_only "
            "and check dir(registrar) for the current attribute name."
        )

    slice_map = {}
    for key, slide_obj in slide_dict.items():
        name_source = None
        for attr in ("src_f", "name", "slide_src_f"):
            val = getattr(slide_obj, attr, None)
            if val:
                name_source = val
                break
        name_source = name_source or key  # fall back to the dict key itself
        try:
            slice_num = get_slice_number(name_source)
        except ValueError:
            logger.warning(
                f"Could not recover a slice number for slide entry "
                f"'{key}' (tried '{name_source}') — skipping. This slide "
                f"will not be matched to any mask."
            )
            continue
        slice_map[slice_num] = slide_obj

    if not slice_map:
        raise RuntimeError(
            "Recovered zero slice-number -> slide mappings from the "
            "registrar. --inspect_only to see what's actually on the "
            "Slide objects and fix build_slice_to_slide_map()."
        )
    return slice_map


# ─────────────────────────────────────────────────────────────────────────────
# WARPING
# ─────────────────────────────────────────────────────────────────────────────
# Candidate keyword spellings for the Slide-level image-warp call, tried in
# order. Different VALIS releases have used slightly different kwarg names;
# this avoids hardcoding one and failing silently on a version mismatch.
WARP_KWARGS_CANDIDATES = [
    dict(img=None, non_rigid=True, crop="overlap", interp_method=cv2.INTER_NEAREST),
    dict(img=None, non_rigid=True, crop=True, interp_method=cv2.INTER_NEAREST),
    dict(img=None, non_rigid=True, crop="overlap", interp_method="nearest"),
]


def warp_mask_with_slide(slide_obj, mask: np.ndarray) -> np.ndarray:
    """Warp a single-channel label mask using this slide's saved transform.

    Tries a few plausible warp_img() call shapes since the exact kwarg
    names vary by VALIS version (see module docstring, point 2).
    """
    if not hasattr(slide_obj, "warp_img"):
        available = [m for m in dir(slide_obj) if "warp" in m.lower()]
        raise AttributeError(
            f"Slide object has no 'warp_img' method. Methods containing "
            f"'warp': {available}. Update WARP_KWARGS_CANDIDATES / the "
            f"call below to match your installed VALIS version."
        )

    last_exc = None
    for kwargs in WARP_KWARGS_CANDIDATES:
        try:
            call_kwargs = dict(kwargs)
            call_kwargs["img"] = mask
            return slide_obj.warp_img(**call_kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
    raise TypeError(
        f"None of the attempted warp_img() call signatures worked. "
        f"Last error: {last_exc}. Run with --inspect_only and check "
        f"help(slide_obj.warp_img) to fix WARP_KWARGS_CANDIDATES."
    )


def process_one_slice(mask_path: str, slide_obj, out_dir: str, qc_dir: str, plot_qc: bool):
    mask = tifffile.imread(mask_path)
    orig_dtype = mask.dtype

    warped = warp_mask_with_slide(slide_obj, mask)
    warped = np.asarray(warped)

    # warp_img on a label image with nearest-neighbor interpolation should
    # preserve integer label IDs exactly; round + cast defensively in case
    # the installed version returns float.
    if not np.issubdtype(warped.dtype, np.integer):
        warped = np.rint(warped)
    warped = warped.astype(orig_dtype)

    base = os.path.splitext(os.path.basename(mask_path))[0]
    out_path = os.path.join(out_dir, f"{base}_warped.tif")
    tifffile.imwrite(out_path, warped)

    if plot_qc:
        _save_qc_panel(mask, warped, mask_path, qc_dir)

    n_labels_before = len(np.unique(mask)) - (1 if 0 in mask else 0)
    n_labels_after = len(np.unique(warped)) - (1 if 0 in warped else 0)
    logger.info(
        f"  {os.path.basename(mask_path)}: {n_labels_before} labels -> "
        f"{n_labels_after} labels after warp -> {out_path}"
    )
    if n_labels_after < n_labels_before:
        logger.warning(
            f"  {os.path.basename(mask_path)}: lost "
            f"{n_labels_before - n_labels_after} label(s) during warp "
            f"(likely cropped out by 'overlap' cropping) — expected for "
            f"cells near the tile edge, but check if the count is large."
        )
    return out_path


def _save_qc_panel(raw_mask, warped_mask, mask_path, qc_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(qc_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(raw_mask > 0, cmap="gray", interpolation="nearest")
    axs[0].set_title(f"Raw mask\n({len(np.unique(raw_mask)) - 1} labels)")
    axs[0].axis("off")
    axs[1].imshow(warped_mask > 0, cmap="gray", interpolation="nearest")
    axs[1].set_title(f"VALIS-warped\n({len(np.unique(warped_mask)) - 1} labels)")
    axs[1].axis("off")
    fig.suptitle(os.path.basename(mask_path))
    fig.tight_layout()
    out_png = os.path.join(
        qc_dir, os.path.splitext(os.path.basename(mask_path))[0] + "_qc.png"
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info(f"VALIS Mask Warping | Core: {TARGET_CORE}")
    logger.info("=" * 60)

    if not os.path.isdir(args.mask_dir):
        logger.error(f"--mask_dir does not exist: {args.mask_dir}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    qc_dir = os.path.join(args.out_dir, "qc")

    registrar = load_registrar(args.deform_dir)

    if args.inspect_only:
        sample = next(iter(registrar.slide_dict.values()))
        logger.info(f"registrar attributes: {sorted(vars(registrar).keys())}")
        logger.info(f"sample Slide object type: {type(sample)}")
        logger.info(f"sample Slide attributes/methods: {sorted(dir(sample))}")
        logger.info("Exiting (--inspect_only). No masks were warped.")
        sys.exit(0)

    slice_to_slide = build_slice_to_slide_map(registrar)

    mask_files = sorted(
        glob.glob(os.path.join(args.mask_dir, "*.tif")) +
        glob.glob(os.path.join(args.mask_dir, "*.tiff")),
        key=get_slice_number,
    )
    if not mask_files:
        logger.error(f"No mask TIFFs found in {args.mask_dir}")
        sys.exit(1)

    jobs = []
    skipped = []
    for mask_path in mask_files:
        slice_num = get_slice_number(mask_path)
        slide_obj = slice_to_slide.get(slice_num)
        if slide_obj is None:
            skipped.append((mask_path, slice_num))
            continue
        jobs.append((mask_path, slide_obj))

    if skipped:
        logger.warning(
            f"{len(skipped)} mask file(s) had no matching registered slide "
            f"and will be SKIPPED (not copied through unwarped, to avoid "
            f"silently mixing registered and unregistered masks downstream): "
            f"{[(os.path.basename(m), s) for m, s in skipped]}"
        )
    if not jobs:
        logger.error("No masks could be matched to a registered slide — nothing to warp.")
        sys.exit(1)

    logger.info(f"Warping {len(jobs)}/{len(mask_files)} masks with {args.workers} workers ...")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one_slice, mask_path, slide_obj, args.out_dir, qc_dir, args.plot_qc): mask_path
            for mask_path, slide_obj in jobs
        }
        for fut in as_completed(futures):
            mask_path = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                failures.append((mask_path, exc))
                logger.error(
                    f"FAILED warping {os.path.basename(mask_path)}: {exc}\n"
                    f"{traceback.format_exc()}"
                )

    if failures:
        logger.error(f"{len(failures)}/{len(jobs)} mask(s) failed to warp.")
        sys.exit(1)

    logger.info(f"Done. {len(jobs)} warped masks written to {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Mirrors valis_register_core2.py's cleanup — safe to call even if
        # the JVM was never actually needed for a pure warp_img() call on
        # already-registered slides, depending on your VALIS version.
        registration.kill_jvm()