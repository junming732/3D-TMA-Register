#!/bin/bash

# --- CONFIGURATION ---
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"
LOG_DIR="log/filter_romav2_bspline"

# --- SETUP ---
source "${VENV_PATH}/bin/activate"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "RoMaV2 + B-spline Sequential Batch"
echo "Cores:       Core_$(printf "%02d" $START) to Core_$(printf "%02d" $END)"
echo "Environment: ${VENV_PATH}"
echo "Log Path:    ${LOG_DIR}/"
echo "Start Time:  $(date)"
echo "============================================"

TOTAL=$((END - START + 1))
DONE=0
FAILED=0
FAILED_CORES=()
BATCH_START=$(date +%s)

for i in $(seq $START $END); do

    CORE_NAME="Core_$(printf "%02d" $i)"
    CORE_START=$(date +%s)

    echo "--------------------------------------------"
    echo "[$(date '+%H:%M:%S')] Processing ${CORE_NAME}... ($((DONE + FAILED + 1))/${TOTAL})"

    python registration/akaze_romav2.py --core_name "${CORE_NAME}" \
        > "${LOG_DIR}/${CORE_NAME}.log" 2>&1

    EXIT_CODE=$?
    CORE_END=$(date +%s)
    CORE_ELAPSED=$(( CORE_END - CORE_START ))
    CORE_MIN=$(( CORE_ELAPSED / 60 ))
    CORE_SEC=$(( CORE_ELAPSED % 60 ))

    if [ $EXIT_CODE -eq 0 ]; then
        DONE=$((DONE + 1))
        # Estimate remaining time based on average so far
        ELAPSED=$(( CORE_END - BATCH_START ))
        AVG=$(( ELAPSED / (DONE + FAILED) ))
        REMAINING=$(( AVG * (TOTAL - DONE - FAILED) ))
        REM_H=$(( REMAINING / 3600 ))
        REM_M=$(( (REMAINING % 3600) / 60 ))
        echo "[SUCCESS] ${CORE_NAME} done in ${CORE_MIN}m${CORE_SEC}s | ~${REM_H}h${REM_M}m remaining"
    else
        FAILED=$((FAILED + 1))
        FAILED_CORES+=("${CORE_NAME}")
        echo "[ERROR]   ${CORE_NAME} failed in ${CORE_MIN}m${CORE_SEC}s. Log: ${LOG_DIR}/${CORE_NAME}.log"
        echo "          --- Last 10 lines ---"
        tail -n 10 "${LOG_DIR}/${CORE_NAME}.log" | sed 's/^/          /'
        echo "          ---------------------"
    fi

done

BATCH_END=$(date +%s)
BATCH_ELAPSED=$(( BATCH_END - BATCH_START ))
BATCH_H=$(( BATCH_ELAPSED / 3600 ))
BATCH_M=$(( (BATCH_ELAPSED % 3600) / 60 ))

echo ""
echo "============================================"
echo "Batch complete at $(date)"
echo "--------------------------------------------"
echo "  Total time: ${BATCH_H}h${BATCH_M}m"
echo "  Processed:  $((DONE + FAILED)) / ${TOTAL}"
echo "  Success:    ${DONE}"
echo "  Failed:     ${FAILED}"
if [ ${#FAILED_CORES[@]} -gt 0 ]; then
    echo "  Failed cores: ${FAILED_CORES[*]}"
fi
echo "============================================"