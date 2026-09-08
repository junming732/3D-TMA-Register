#!/bin/bash

# =============================================================================
# run_all_3d_analysis.sh
# Step 4: 3D connected-component analysis across registered CellPose masks
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CORE_NUMS=(9)

# Global 3D analysis flags
MIN_SLICES=1
MIN_CONFIRMED=2
MIN_IOU=0.15
MIN_INTENSITY_FRAC=0.3

# DAPI-specific linking constraints
MAX_SLICES=5
MIN_AREA_PX=200
MIN_OVERLAP=30

COLOC_CHANNEL=""
COLOC_RADIUS_UM=50

# Pipeline matching configuration
REG_VARIANT="romav2" # romav2 Change to "valis" to write the QC baseline
if [ "${REG_VARIANT}" == "valis" ]; then
    # Replace these with your actual Valis folder names
    REG_DIR_NAME="VALIS_Filter_Eval" 
    INPUT_DIR_NAME="CellPose_DAPI_Warped_Valis"
    OUTPUT_DIR_NAME="CellPose_DAPI_3D_Valis"
    DENOISED_DIR_NAME="Denoised_Valis"
elif [ "${REG_VARIANT}" == "romav2" ]; then
    REG_DIR_NAME="Filter_AKAZE_RoMaV2_Linear_Warp_map_multi_channel_clahe_visual_additive_3ch"
    INPUT_DIR_NAME="CellPose_DAPI_Warped_RomaV2_clahe_visual_additive_3ch"
    OUTPUT_DIR_NAME="CellPose_DAPI_3D_RomaV2_clahe_visual_additive_3ch"
    DENOISED_DIR_NAME="Denoised_RomaV2_clahe_visual_additive_3ch"
fi
COLOC_DIR_NAME=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ANALYSIS_SCRIPT="${PROJECT_ROOT}/spatial_analysis/link_3d_cells.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_3D="${LOG_ROOT}/3d_linkage_${REG_VARIANT}"

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
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

TOTAL=${#CORE_NUMS[@]}
DONE=0
FAIL=0
SKIP=0

declare -A CORE_STATUS

echo "============================================================"
echo "  3D Cell Analysis Pipeline (DAPI)"
echo "  Cores     : $(printf "Core_%02d " "${CORE_NUMS[@]}")"
echo "  Variant   : ${REG_VARIANT}"
echo "  Start time: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
IDX=0
for i in "${CORE_NUMS[@]}"; do
    CORE_NAME="Core_$(printf "%02d" $i)"
    IDX=$((IDX + 1))

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')]  ${CORE_NAME}  (${IDX}/${TOTAL})"
    echo "------------------------------------------------------------"

    WARPED_DIR="${DATASPACE}${INPUT_DIR_NAME}/${CORE_NAME}"

    if [ ! -d "${WARPED_DIR}" ]; then
        echo "  [SKIP] No warped mask directory at ${WARPED_DIR}"
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_MASKS"
        continue
    fi

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

    # 1. Determine if this is the reference pipeline
    QC_REF_FLAG=""
    if [ "${REG_VARIANT}" == "valis" ]; then
        QC_REF_FLAG="--set_qc_reference"
    fi

    # 2. Determine pipeline kind and raw-space mapping directory based on Snakefile logic
    QC_FLAGS=""
    if [ "${REG_VARIANT}" == "valis" ] || [ "${REG_VARIANT}" == "romav2" ]; then
        QC_PIPELINE_KIND="${REG_VARIANT}"
        
        if [ "${REG_VARIANT}" == "valis" ]; then
            QC_TRANSFORM_DIR="${DATASPACE}${REG_DIR_NAME}/${CORE_NAME}/${CORE_NAME}/point_transforms"
        else
            QC_TRANSFORM_DIR="${DATASPACE}${REG_DIR_NAME}/${CORE_NAME}/deformation_maps"
        fi
        
        QC_FLAGS="--qc_pipeline_kind ${QC_PIPELINE_KIND} --qc_transform_dir ${QC_TRANSFORM_DIR}"
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
        ${QC_REF_FLAG} \
        ${QC_FLAGS} \
        ${COLOC_FLAG} \
        ${COLOC_DIR_FLAG} \
        > "${LOG_3D}/${CORE_NAME}.log" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        FAIL=$((FAIL + 1))
        CORE_STATUS[$CORE_NAME]="FAIL"
        echo "  [FAIL] 3D analysis failed."
        echo "         Log: ${LOG_3D}/${CORE_NAME}.log"
    else
        DONE=$((DONE + 1))
        CORE_STATUS[$CORE_NAME]="OK"
        N_CELLS=$(grep "Final 3D cell count" "${LOG_3D}/${CORE_NAME}.log" | tail -1 | grep -o '[0-9]*$')
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
for i in "${CORE_NUMS[@]}"; do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs  : ${LOG_3D}/"
echo "  Output: ${DATASPACE}${OUTPUT_DIR_NAME}/<CORE_NAME>/"
echo "============================================================"