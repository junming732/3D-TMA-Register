#!/bin/bash

# =============================================================================
# run_step2_cellpose.sh
# Step 2: CellPose segmentation
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────
START=5
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# CellPose extra flags
GPU_FLAGS="--use_gpu"
CELLPOSE_FLAGS="--plot_qc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CP_SCRIPT="${PROJECT_ROOT}/registration/cellpose_segmentation.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_CP="${LOG_ROOT}/cellpose"

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
source "${VENV_PATH}/bin/activate"

# Derive DATASPACE directly from config.py so this script never drifts out of sync
DATASPACE="$(python -c "import sys; sys.path.insert(0,'${PROJECT_ROOT}'); import config; print(config.DATASPACE)")"
if [ -z "${DATASPACE}" ]; then
    echo "[ERROR] Could not read DATASPACE from config.py — aborting."
    exit 1
fi
echo "  DATASPACE : ${DATASPACE}"

mkdir -p "${LOG_CP}"

TOTAL=$((END - START + 1))
DONE_CP=0;   FAIL_CP=0

declare -A CORE_STATUS

echo "============================================================"
echo "  CellPose Pipeline: Segmentation"
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

    echo "  [1/1] CellPose — segmenting DAPI..."
    python "${CP_SCRIPT}" \
        --core_name "${CORE_NAME}" \
        ${CELLPOSE_FLAGS} \
        ${GPU_FLAGS} \
        > "${LOG_CP}/${CORE_NAME}_DAPI.log" 2>&1
    CP_EXIT=$?

    if [ $CP_EXIT -ne 0 ]; then
        FAIL_CP=$((FAIL_CP + 1))
        CORE_STATUS[$CORE_NAME]="FAIL_CP"
        echo "  [FAIL] CellPose failed."
        echo "         Log: ${LOG_CP}/${CORE_NAME}_DAPI.log"
        echo "         --- last 10 lines ---"
        tail -n 10 "${LOG_CP}/${CORE_NAME}_DAPI.log" | sed 's/^/         /'
        echo "         ---------------------"
    else
        DONE_CP=$((DONE_CP + 1))
        CORE_STATUS[$CORE_NAME]="OK"
        echo "  [OK]   CellPose complete."
    fi

done

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  CellPose pipeline complete — $(date)"
echo "------------------------------------------------------------"
echo "  Cores processed : ${TOTAL}"
printf "  CellPose (DAPI) : %d OK  |  %d FAILED\n" $DONE_CP $FAIL_CP
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
echo "============================================================"