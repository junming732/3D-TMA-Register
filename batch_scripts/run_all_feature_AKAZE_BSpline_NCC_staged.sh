#!/bin/bash

# --- CONFIGURATION ---
# Range of cores to process (Core_01 to Core_30)
START=1
END=30

# YOUR ACTUAL ENVIRONMENT PATH
VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# LOG DIRECTORY
LOG_DIR="log/feature_AKAZE_BSpline_NCC_staged"

# 1. Activate environment
source "${VENV_PATH}/bin/activate"

# 2. Create log directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

echo "============================================"
echo "Starting Staged Sequential Processing: Core_$(printf "%02d" $START) to Core_$(printf "%02d" $END)"
echo "  Stage 1: AKAZE affine"
echo "  Stage 2: B-spline NCC elastic refinement"
echo "Environment: ${VENV_PATH}"
echo "Log Path:    ${LOG_DIR}/"
echo "Start Time:  $(date)"
echo "============================================"

TOTAL=$((END - START + 1))
DONE=0
FAILED_AKAZE=0
FAILED_BSPLINE=0

for i in $(seq $START $END); do

    CORE_NAME="Core_$(printf "%02d" $i)"
    LOG_AKAZE="${LOG_DIR}/${CORE_NAME}_stage1_AKAZE.log"
    LOG_BSPLINE="${LOG_DIR}/${CORE_NAME}_stage2_BSpline.log"

    echo "--------------------------------------------"
    echo "[$(date '+%H:%M:%S')] Processing ${CORE_NAME} ($((DONE + FAILED_AKAZE + FAILED_BSPLINE + 1))/${TOTAL})"

    # ── Stage 1: AKAZE affine ────────────────────────────────────────────────
    echo "[$(date '+%H:%M:%S')]   Stage 1 — AKAZE affine..."
    python registration/feature_registration_AKAZE.py --core_name "${CORE_NAME}" \
        > "${LOG_AKAZE}" 2>&1
    EXIT_AKAZE=$?

    if [ $EXIT_AKAZE -ne 0 ]; then
        FAILED_AKAZE=$((FAILED_AKAZE + 1))
        echo "[ERROR]   ${CORE_NAME} Stage 1 (AKAZE) failed — skipping Stage 2."
        echo "          Check log: ${LOG_AKAZE}"
        echo "          --- Last 10 lines ---"
        tail -n 10 "${LOG_AKAZE}" | sed 's/^/          /'
        echo "          ---------------------"
        continue   # skip B-spline for this core
    fi

    echo "[$(date '+%H:%M:%S')]   Stage 1 — AKAZE complete."

    # ── Stage 2: B-spline NCC elastic ───────────────────────────────────────
    echo "[$(date '+%H:%M:%S')]   Stage 2 — B-spline NCC..."
    python registration/feature_registration_BSpline_NCC.py --core_name "${CORE_NAME}" \
        > "${LOG_BSPLINE}" 2>&1
    EXIT_BSPLINE=$?

    if [ $EXIT_BSPLINE -ne 0 ]; then
        FAILED_BSPLINE=$((FAILED_BSPLINE + 1))
        echo "[ERROR]   ${CORE_NAME} Stage 2 (B-spline) failed."
        echo "          Check log: ${LOG_BSPLINE}"
        echo "          --- Last 10 lines ---"
        tail -n 10 "${LOG_BSPLINE}" | sed 's/^/          /'
        echo "          ---------------------"
        continue
    fi

    DONE=$((DONE + 1))
    echo "[SUCCESS] ${CORE_NAME} — both stages complete."

done

echo ""
echo "============================================"
echo "All jobs finished at $(date)"
echo "--------------------------------------------"
echo "  Cores attempted:        $((DONE + FAILED_AKAZE + FAILED_BSPLINE)) / ${TOTAL}"
echo "  Both stages success:    ${DONE}"
echo "  Failed at Stage 1:      ${FAILED_AKAZE}"
echo "  Failed at Stage 2:      ${FAILED_BSPLINE}"
echo "============================================"