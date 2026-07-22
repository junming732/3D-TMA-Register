#!/bin/bash

# =============================================================================
# run_cellpose.sh
# Steps 2 & 3: CellPose segmentation → CellPose mask warping
#
# Requires registration to have already been run (run_registration.sh).
# A core with no deformation maps is skipped at Step 3.
# A core that fails at Step 2 is skipped at Step 3.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────
START=1
END=30

# CellPose extra flags
GPU_FLAGS="--use_gpu"
CELLPOSE_FLAGS="--plot_qc"

# Warp script extra flags
WARP_FLAGS="--plot_qc"

# Registration variant to read deformation maps from — must match the
# INPUT_DIR_NAME used in run_all_denoising.sh / denoise_volume.py so mask
# warping and volume denoising stay aligned to the same registration run.
DEFORM_DIR_NAME="Filter_AKAZE_TissueMask_BSpline"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CP_SCRIPT="${PROJECT_ROOT}/spatial_analysis/cellpose_segmentation.py"
WARP_SCRIPT="${PROJECT_ROOT}/spatial_analysis/warp_cellpose_masks.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_CP="${LOG_ROOT}/cellpose"
LOG_WARP="${LOG_ROOT}/warp"

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
# Read VENV_PATH with system python3, BEFORE activating any venv
VENV_PATH="$(python3 -c "import sys; sys.path.insert(0,'${PROJECT_ROOT}'); import config; print(config.VENV_PATH)")"
if [ -z "${VENV_PATH}" ]; then
    echo "[ERROR] Could not read VENV_PATH from config.py -- aborting."
    exit 1
fi

source "${VENV_PATH}/bin/activate"

# Derive DATASPACE directly from config.py so this script never drifts out of sync
DATASPACE="$(python -c "import sys; sys.path.insert(0,'${PROJECT_ROOT}'); import config; print(config.DATASPACE)")"
if [ -z "${DATASPACE}" ]; then
    echo "[ERROR] Could not read DATASPACE from config.py — aborting."
    exit 1
fi
echo "  DATASPACE : ${DATASPACE}"

mkdir -p "${LOG_CP}" "${LOG_WARP}"

TOTAL=$((END - START + 1))
DONE_CP=0;   FAIL_CP=0
DONE_WARP=0; FAIL_WARP=0

declare -A CORE_STATUS

echo "============================================================"
echo "  CellPose Pipeline: Segmentation → Mask Warp"
echo "  Cores     : Core_$(printf "%02d" $START) → Core_$(printf "%02d" $END)"
echo "  Channel   : DAPI (Hardcoded)"
echo "  Start time: $(date)"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
for i in $(seq $START $END); do

    CORE_NAME="Core_$(printf "%02d" $i)"
    IDX=$((i - START + 1))

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')]  ${CORE_NAME}  (${IDX}/${TOTAL})"
    echo "------------------------------------------------------------"

    # ── STEP 2: CellPose segmentation (DAPI only) ───────────────
    CP_OK=1

    echo "  [1/2] CellPose — segmenting DAPI..."
    python "${CP_SCRIPT}" \
        --core_name "${CORE_NAME}" \
        ${CELLPOSE_FLAGS} \
        ${GPU_FLAGS} \
        > "${LOG_CP}/${CORE_NAME}_DAPI.log" 2>&1
    CP_EXIT=$?

    if [ $CP_EXIT -ne 0 ]; then
        CP_OK=0
        FAIL_CP=$((FAIL_CP + 1))
        CORE_STATUS[$CORE_NAME]="FAIL_CP"
        echo "  [FAIL] CellPose failed — skipping warp for this core."
        echo "         Log: ${LOG_CP}/${CORE_NAME}_DAPI.log"
        echo "         --- last 10 lines ---"
        tail -n 10 "${LOG_CP}/${CORE_NAME}_DAPI.log" | sed 's/^/         /'
        echo "         ---------------------"
    else
        DONE_CP=$((DONE_CP + 1))
        echo "  [OK]   CellPose complete."
    fi

    # ── STEP 3: Warp masks (only if CellPose succeeded) ─────
    WARP_OK=1

    if [ $CP_OK -eq 1 ]; then
        MASK_DIR="${DATASPACE}CellPose_DAPI/${CORE_NAME}"
        DEFORM_DIR="${DATASPACE}${DEFORM_DIR_NAME}/${CORE_NAME}/deformation_maps"
        OUT_DIR="${DATASPACE}CellPose_DAPI_Warped/${CORE_NAME}"

        if [ ! -d "${MASK_DIR}" ]; then
            echo "  [SKIP] Warp: no mask directory found at ${MASK_DIR}"
            WARP_OK=0
        elif [ ! -d "${DEFORM_DIR}" ]; then
            echo "  [SKIP] Warp: no deformation maps at ${DEFORM_DIR} — run registration first."
            WARP_OK=0
        else
            echo "  [2/2] Warp masks — DAPI..."
            python "${WARP_SCRIPT}" \
                --core_name  "${CORE_NAME}" \
                --mask_dir   "${MASK_DIR}" \
                --deform_dir "${DEFORM_DIR}" \
                --out_dir    "${OUT_DIR}" \
                ${WARP_FLAGS} \
                > "${LOG_WARP}/${CORE_NAME}_DAPI.log" 2>&1
            WARP_EXIT=$?

            if [ $WARP_EXIT -ne 0 ]; then
                WARP_OK=0
                FAIL_WARP=$((FAIL_WARP + 1))
                CORE_STATUS[$CORE_NAME]="FAIL_WARP"
                echo "  [FAIL] Warp failed."
                echo "         Log: ${LOG_WARP}/${CORE_NAME}_DAPI.log"
                echo "         --- last 10 lines ---"
                tail -n 10 "${LOG_WARP}/${CORE_NAME}_DAPI.log" | sed 's/^/         /'
                echo "         ---------------------"
            else
                DONE_WARP=$((DONE_WARP + 1))
                CORE_STATUS[$CORE_NAME]="OK"
                echo "  [OK]   Warp complete."
            fi
        fi
    fi

done

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
MAX_CP=$TOTAL
MAX_WARP=$TOTAL

echo ""
echo "============================================================"
echo "  CellPose pipeline complete — $(date)"
echo "------------------------------------------------------------"
echo "  Cores processed : ${TOTAL}"
printf "  CellPose (DAPI) : %d OK  |  %d FAILED\n" $DONE_CP $FAIL_CP
printf "  Mask warp (DAPI): %d OK  |  %d FAILED\n" $DONE_WARP $FAIL_WARP
echo "------------------------------------------------------------"

echo "  Per-core status:"
for i in $(seq $START $END); do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs:"
echo "    CellPose : ${LOG_CP}/"
echo "    Warp     : ${LOG_WARP}/"
echo "============================================================"