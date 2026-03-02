#!/bin/bash

# --- CONFIGURATION ---
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"
LOG_DIR="log/Exploration_preprocessing"

# 1. Activate environment
source "${VENV_PATH}/bin/activate"

# 2. Create log directory
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

echo "============================================"
echo "Starting Sequential Exploration: Core_$(printf "%02d" $START) to Core_$(printf "%02d" $END)"
echo "Script:      registration/compare_preprocessing_feature.py"
echo "Input:       TMA_Cores_Grouped_Rotate/"
echo "Output:      Exploration_preprocessing/"
echo "Log Path:    ${LOG_DIR}/"
echo "Start Time:  $(date)"
echo "============================================"

TOTAL=$((END - START + 1))
DONE=0
FAILED=0

for i in $(seq $START $END); do

    CORE_NAME="Core_$(printf "%02d" $i)"

    echo "--------------------------------------------"
    echo "[$(date '+%H:%M:%S')] Evaluating ${CORE_NAME}... ($((DONE + FAILED + 1))/${TOTAL})"

    # Execute the exploration script
    python experiments/compare_preprocessing_feature.py --core_name "${CORE_NAME}" \
        > "${LOG_DIR}/${CORE_NAME}.log" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        DONE=$((DONE + 1))
        echo "[SUCCESS] ${CORE_NAME} completed."
    else
        FAILED=$((FAILED + 1))
        echo "[ERROR]   ${CORE_NAME} failed. Check log: ${LOG_DIR}/${CORE_NAME}.log"
        echo "          --- Last 10 lines of log ---"
        tail -n 10 "${LOG_DIR}/${CORE_NAME}.log" | sed 's/^/          /'
        echo "          ----------------------------"
    fi

done

echo ""
echo "============================================"
echo "All evaluation jobs finished at $(date)"
echo "--------------------------------------------"
echo "  Processed:  $((DONE + FAILED)) / ${TOTAL}"
echo "  Success:    ${DONE}"
echo "  Failed:     ${FAILED}"
echo "============================================"