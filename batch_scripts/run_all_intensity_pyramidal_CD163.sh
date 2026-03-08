#!/bin/bash

# --- CONFIGURATION ---
# Range of cores to process (Core_01 to Core_30)
START=1
END=30

# YOUR ACTUAL ENVIRONMENT PATH
VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# LOG DIRECTORY
LOG_DIR="log/Intensity_pyramidal_CD163"

# 1. Activate environment
source "${VENV_PATH}/bin/activate"

# 2. Create log directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

echo "============================================"
echo "Starting Sequential Processing (intensity): Core_$(printf "%02d" $START) to Core_$(printf "%02d" $END)"
echo "Environment: ${VENV_PATH}"
echo "Log Path:    ${LOG_DIR}/"
echo "Start Time:  $(date)"
echo "============================================"

TOTAL=$((END - START + 1))
DONE=0
FAILED=0
AKAZE_USED=()

for i in $(seq $START $END); do

    CORE_NAME="Core_$(printf "%02d" $i)"

    echo "--------------------------------------------"
    echo "[$(date '+%H:%M:%S')] Processing ${CORE_NAME}... ($((DONE + FAILED + 1))/${TOTAL})"

    python registration/intensity_pyramidal_CD163.py --core_name "${CORE_NAME}" \
        > "${LOG_DIR}/${CORE_NAME}.log" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        DONE=$((DONE + 1))
        echo "[SUCCESS] ${CORE_NAME} completed."
    else
        FAILED=$((FAILED + 1))
        echo "[ERROR]   ${CORE_NAME} failed. Check log: ${LOG_DIR}/${CORE_NAME}.log"
        # Print the last 10 lines of the log to give immediate context
        echo "          --- Last 10 lines of log ---"
        tail -n 10 "${LOG_DIR}/${CORE_NAME}.log" | sed 's/^/          /'
        echo "          ----------------------------"
    fi

done

echo ""
echo "============================================"
echo "All jobs finished at $(date)"
echo "--------------------------------------------"
echo "  Processed:  $((DONE + FAILED)) / ${TOTAL}"
echo "  Success:    ${DONE}"
echo "  Failed:     ${FAILED}"
echo "============================================"