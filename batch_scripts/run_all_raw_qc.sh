#!/bin/bash

# --- CONFIGURATION ---
START=1
END=30

# Preserve existing environment path
VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Isolated log directory for raw visualization
LOG_DIR="log/qc_raw"
CHANNEL_IDX=6

# 1. Activate environment
source "${VENV_PATH}/bin/activate"

# 2. Create log directory if it does not exist
if [ ! -d "$LOG_DIR" ]; then
    echo "Initializing log directory: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

echo "============================================"
echo "Starting Raw QC Batch Processing: Core_$(printf "%02d" $START) to Core_$(printf "%02d" $END)"
echo "Script:      visualize_raw_alignment.py"
echo "Target Ch:   ${CHANNEL_IDX}"
echo "Environment: ${VENV_PATH}"
echo "Log Path:    ${LOG_DIR}/"
echo "Start Time:  $(date)"
echo "============================================"

# Loop through core indices
for i in $(seq $START $END); do
    
    # Format the core name to match directory structure (Core_01, Core_02...)
    CORE_NAME="Core_$(printf "%02d" $i)"
    
    echo "--------------------------------------------"
    echo "[$(date)] Inspecting ${CORE_NAME}..."
    
    # Execute the Python script and redirect output to the designated log file
    python registration/visualize_raw_alignment.py --core_name "${CORE_NAME}" --channel_idx ${CHANNEL_IDX} > "${LOG_DIR}/${CORE_NAME}.log" 2>&1
    
    # Verify execution status
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] ${CORE_NAME} montage generated."
    else
        echo "[FAILURE] ${CORE_NAME} encountered an error. Review ${LOG_DIR}/${CORE_NAME}.log for tracebacks."
    fi
    
    # Allow system resources to settle before loading the next core into memory
    sleep 3
done

echo "============================================"
echo "Batch QC visualization completed at $(date)"
echo "============================================"