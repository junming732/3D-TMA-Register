#!/bin/bash

# =============================================================================
# run_step3b_cellpose3d.sh
# Stage 3b: Native 3D CellPose segmentation on denoised volumes
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────
CORE_NUMS=(9 16 19)   # <-- explicit list of core numbers to run

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Pipeline tag and directory names 
TAG="RomaV2_clahe_visual_additive_2ch"
DENOISE_DIR_NAME="Denoised_${TAG}"
CELLPOSE3D_DIR_NAME="CellPose3D_${TAG}"

# CellPose 3D parameters matching Snakefile defaults
FLOW_THRESHOLD="0.6"
CELLPROB_THRESHOLD="0.0"
QC_FLAGS="--skip-qc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CP3D_SCRIPT="${PROJECT_ROOT}/spatial_analysis/test_3d_cellpose.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_CP3D="${LOG_ROOT}/cellpose3d"

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
source "${VENV_PATH}/bin/activate"

# Derive DATASPACE directly from config.py
DATASPACE="$(python -c "import sys; sys.path.insert(0,'${PROJECT_ROOT}'); import config; print(config.DATASPACE)")"
if [ -z "${DATASPACE}" ]; then
    echo "[ERROR] Could not read DATASPACE from config.py — aborting."
    exit 1
fi
echo "  DATASPACE : ${DATASPACE}"

mkdir -p "${LOG_CP3D}"

TOTAL=${#CORE_NUMS[@]}
DONE_CP3D=0; FAIL_CP3D=0
SKIP=0

declare -A CORE_STATUS

echo "============================================================"
echo "  CellPose 3D Pipeline: Native 3D Segmentation"
echo "  Cores     : $(printf "Core_%02d " "${CORE_NUMS[@]}")"
echo "  Input Dir : ${DENOISE_DIR_NAME}"
echo "  Output Dir: ${CELLPOSE3D_DIR_NAME}"
echo "  Start time: $(date)"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
IDX=0
for i in "${CORE_NUMS[@]}"; do

    CORE_NAME="Core_$(printf "%02d" $i)"
    IDX=$((IDX + 1))

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')]  ${CORE_NAME}  (${IDX}/${TOTAL})"
    echo "------------------------------------------------------------"

    INPUT_FILE="${DATASPACE}/${DENOISE_DIR_NAME}/${CORE_NAME}/${CORE_NAME}_denoised.ome.tif"
    OUTPUT_DIR="${DATASPACE}/${CELLPOSE3D_DIR_NAME}/${CORE_NAME}"
    OUTPUT_FILE="${OUTPUT_DIR}/${CORE_NAME}_3D_Cellpose_Masks.tif"

    if [ ! -f "${INPUT_FILE}" ]; then
        echo "  [SKIP] Input volume not found: ${INPUT_FILE}"
        CORE_STATUS[$CORE_NAME]="SKIP_NO_INPUT"
        SKIP=$((SKIP + 1))
        continue
    fi

    mkdir -p "${OUTPUT_DIR}"

    echo "  [1/1] CellPose 3D — segmenting volume..."
    
    python -u "${CP3D_SCRIPT}" \
        --input "${INPUT_FILE}" \
        --output "${OUTPUT_FILE}" \
        --flow-threshold "${FLOW_THRESHOLD}" \
        --cellprob-threshold "${CELLPROB_THRESHOLD}" \
        ${QC_FLAGS} \
        > "${LOG_CP3D}/${CORE_NAME}_3D.log" 2>&1
        
    CP3D_EXIT=$?

    if [ $CP3D_EXIT -ne 0 ]; then
        FAIL_CP3D=$((FAIL_CP3D + 1))
        CORE_STATUS[$CORE_NAME]="FAIL"
        echo "  [FAIL] CellPose 3D failed."
        echo "         Log: ${LOG_CP3D}/${CORE_NAME}_3D.log"
        echo "         --- last 10 lines ---"
        tail -n 10 "${LOG_CP3D}/${CORE_NAME}_3D.log" | sed 's/^/         /'
        echo "         ---------------------"
    else
        DONE_CP3D=$((DONE_CP3D + 1))
        CORE_STATUS[$CORE_NAME]="OK"
        echo "  [OK]   CellPose 3D complete."
    fi

done

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  CellPose 3D pipeline complete — $(date)"
echo "------------------------------------------------------------"
echo "  Cores targeted : ${TOTAL}"
printf "  CellPose 3D    : %d OK  |  %d FAILED  |  %d SKIPPED\n" $DONE_CP3D $FAIL_CP3D $SKIP
echo "------------------------------------------------------------"

echo "  Per-core status:"
for i in "${CORE_NUMS[@]}"; do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done
echo "============================================================"