#!/bin/bash

# =============================================================================
# run_all_3d_analysis.sh
# Step 4: 3D connected-component analysis across registered CellPose masks
#
# Requires Steps 1-3 (registration -> CellPose -> warp) to have already run.
# CellPose (and therefore this step) only ever runs on DAPI.
# Reads warped DAPI masks from ${INPUT_DIR_NAME}/<CORE_NAME>/ (see CONFIGURATION below)
# Writes 3D label volumes + stats to ${OUTPUT_DIR_NAME}/<CORE_NAME>/
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
START=1
END=30

# =============================================================================
# ARGUMENT GUIDE — what each flag controls and why they do NOT conflict
# =============================================================================
# These filters act at DIFFERENT stages of the pipeline:
#
#  Stage 1 — LINK ACCEPTANCE (per adjacent-slice pair):
#    --min_overlap         : raw pixel count gate.  First rough filter.
#    --min_overlap_frac    : overlap / smaller_area. Normalises for size diff.
#    --min_iou             : overlap / union.  Stricter; penalises area mismatch.
#    --min_intensity_frac  : mean DAPI of each slice >= fraction of cell peak.
#                            Catches dark/empty mid-stack CellPose masks.
#    -> These work TOGETHER. A link must pass ALL active thresholds.
#       They are complementary, not contradictory.
#
#  Stage 2 — COMPONENT PRUNING (after graph is built):
#    --max_slices          : severs weakest edge if z_max - z_min + 1 > limit.
#                            Controls Z-SPAN (height of the 3D cell).
#    --max_segments_per_z  : rejects cells with >N masks on a single Z-slice.
#                            Catches CellPose over-segmentation (the "8-panel
#                            but span=3" bug shown in cell_105412).
#    -> These work on the component AFTER all links are accepted.
#
#  Stage 3 — OUTPUT FILTERING (keep / discard whole 3D cells):
#    --min_slices          : minimum z_span to keep a cell (removes singletons).
#    --min_area_px         : minimum peak 2D area.
#    --min_confirmed       : minimum z_span to count as "confirmed" in QC.
# =============================================================================

# Global 3D analysis flags
MIN_SLICES=1
MIN_CONFIRMED=2
MIN_IOU=0.15
MIN_INTENSITY_FRAC=0.3

# DAPI-specific linking constraints (nuclei: ~4.5 um sections, ~10 um diameter)
MAX_SLICES=5       # max Z-span for a nucleus (2-4 slices typical)
MIN_AREA_PX=200    # consistent with CellPose min_size for DAPI
MIN_OVERLAP=30     # larger nuclei need more overlap to confirm a true link

# Co-localisation: channel to compare DAPI-linked 3D cells against, or leave
# empty to skip. This is a lookup against another channel's already-computed
# 3D stats — it does not imply running CellPose on that channel here.
COLOC_CHANNEL=""
COLOC_RADIUS_UM=50

# Registration-variant-dependent folders — change these to point at a
# different registration run without editing link_3d_cells.py.
INPUT_DIR_NAME="CellPose_DAPI_Warped_Bspline"
OUTPUT_DIR_NAME="CellPose_DAPI_3D_Bspline"
DENOISED_DIR_NAME="Denoised_bspline"
# Leave empty to default to CellPose_<COLOC_CHANNEL>_3D_Bspline
COLOC_DIR_NAME=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ANALYSIS_SCRIPT="${PROJECT_ROOT}/registration/link_3d_cells.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_3D="${LOG_ROOT}/3d_linkage_bspline"

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
# Read VENV_PATH with system python3, BEFORE activating any venv
VENV_PATH="$(python3 -c "import sys; sys.path.insert(0,'${PROJECT_ROOT}'); import config; print(config.VENV_PATH)")"
if [ -z "${VENV_PATH}" ]; then
    echo "[ERROR] Could not read VENV_PATH from config.py -- aborting."
    exit 1
fi

source "${VENV_PATH}/bin/activate"

DATASPACE="$(python -c "import sys; sys.path.insert(0,'${PROJECT_ROOT}'); import config; print(config.DATASPACE)")"
if [ -z "${DATASPACE}" ]; then
    echo "[ERROR] Could not read DATASPACE from config.py -- aborting."
    exit 1
fi
echo "  DATASPACE : ${DATASPACE}"

mkdir -p "${LOG_3D}"

TOTAL=$((END - START + 1))
DONE=0
FAIL=0
SKIP=0

declare -A CORE_STATUS

echo "============================================================"
echo "  3D Cell Analysis Pipeline (DAPI)"
echo "  Cores     : Core_$(printf "%02d" $START) -> Core_$(printf "%02d" $END)"
echo "  Max slices: ${MAX_SLICES}"
echo "  Start time: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
for i in $(seq $START $END); do

    CORE_NAME="Core_$(printf "%02d" $i)"
    IDX=$((i - START + 1))

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')]  ${CORE_NAME}  (${IDX}/${TOTAL})"
    echo "------------------------------------------------------------"

    WARPED_DIR="${DATASPACE}${INPUT_DIR_NAME}/${CORE_NAME}"

    # Skip if warped masks do not exist yet
    if [ ! -d "${WARPED_DIR}" ]; then
        echo "  [SKIP] No warped mask directory at ${WARPED_DIR}"
        echo "         Run registration + CellPose + warp first."
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_MASKS"
        continue
    fi

    # Skip if no warped mask files inside
    N_MASKS=$(find "${WARPED_DIR}" -name "*DAPI*_warped.tif" | wc -l)
    if [ "${N_MASKS}" -eq 0 ]; then
        echo "  [SKIP] Directory exists but contains no *_warped.tif files."
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_MASKS"
        continue
    fi

    echo "  [RUN] 3D analysis (${N_MASKS} warped slices)..."

    COLOC_FLAG=""
    if [ -n "${COLOC_CHANNEL}" ] && [ "${COLOC_CHANNEL}" != "DAPI" ]; then
        COLOC_FLAG="--coloc_channel ${COLOC_CHANNEL}"
    fi

    COLOC_DIR_FLAG=""
    if [ -n "${COLOC_DIR_NAME}" ]; then
        COLOC_DIR_FLAG="--coloc_dir_name ${COLOC_DIR_NAME}"
    fi

    python "${ANALYSIS_SCRIPT}" \
        --core_name "${CORE_NAME}" \
        --plot_qc \
        --min_slices ${MIN_SLICES} \
        --max_slices ${MAX_SLICES} \
        --min_area_px ${MIN_AREA_PX} \
        --min_overlap ${MIN_OVERLAP} \
        --min_confirmed ${MIN_CONFIRMED} \
        --coloc_radius_um ${COLOC_RADIUS_UM} \
        --min_iou ${MIN_IOU} \
        --min_intensity_frac ${MIN_INTENSITY_FRAC} \
        --input_dir_name    "${INPUT_DIR_NAME}" \
        --output_dir_name   "${OUTPUT_DIR_NAME}" \
        --denoised_dir_name "${DENOISED_DIR_NAME}" \
        ${COLOC_FLAG} \
        ${COLOC_DIR_FLAG} \
        > "${LOG_3D}/${CORE_NAME}.log" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        FAIL=$((FAIL + 1))
        CORE_STATUS[$CORE_NAME]="FAIL"
        echo "  [FAIL] 3D analysis failed."
        echo "         Log: ${LOG_3D}/${CORE_NAME}.log"
        echo "         --- last 10 lines ---"
        tail -n 10 "${LOG_3D}/${CORE_NAME}.log" | sed 's/^/         /'
        echo "         ---------------------"
    else
        DONE=$((DONE + 1))
        CORE_STATUS[$CORE_NAME]="OK"
        # Extract cell count from log for quick inline summary
        N_CELLS=$(grep "Final 3D cell count" "${LOG_3D}/${CORE_NAME}.log" \
                  | tail -1 | grep -o '[0-9]*$')
        echo "  [OK]   3D analysis complete -- ${N_CELLS:-?} 3D cells."
    fi

done

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  3D analysis complete -- $(date)"
echo "------------------------------------------------------------"
echo "  Cores processed : ${TOTAL}"
printf "  Results         : %d OK  |  %d FAILED  |  %d SKIPPED  (of %d)\n" \
       $DONE $FAIL $SKIP $TOTAL
echo "------------------------------------------------------------"

echo "  Per-core status:"
for i in $(seq $START $END); do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs  : ${LOG_3D}/"
echo "  Output: ${DATASPACE}${OUTPUT_DIR_NAME}/<CORE_NAME>/"
echo "============================================================"