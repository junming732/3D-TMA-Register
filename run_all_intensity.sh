#!/bin/bash

# --- CONFIGURATION ---
# Range of cores to process (Core_01 to Core_30)
START=1
END=30

# YOUR ACTUAL ENVIRONMENT PATH
VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# LOG DIRECTORY
LOG_DIR="log/intensity"

# 1. Activate environment
source "${VENV_PATH}/bin/activate"

# 2. Create log directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

echo "============================================"
echo "Starting Sequential Processing: Core_$(printf "%02d" $START) to Core_$(printf "%02d" $END)"
echo "Script:      Register_Single_Core.py"
echo "Environment: ${VENV_PATH}"
echo "Log Path:    ${LOG_DIR}/"
echo "Start Time:  $(date)"
echo "============================================"

# Loop through 1 to 30
for i in $(seq $START $END); do
    
    # Format the core name with underscore and 2 digits (Core_01, Core_02...)
    CORE_NAME="Core_$(printf "%02d" $i)"
    
    echo "--------------------------------------------"
    echo "[$(date)] Processing ${CORE_NAME}..."
    
    # Run the Python script
    # Logs are saved to log/intensity/Core_01.log, etc.
    python intensity_based_DAPI.py --core_name "${CORE_NAME}" > "${LOG_DIR}/${CORE_NAME}.log" 2>&1
    
    # Check status
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] ${CORE_NAME} finished."
    else
        echo "[FAILURE] ${CORE_NAME} failed. See ${LOG_DIR}/${CORE_NAME}.log"
    fi
    
    # Pause to let memory flush
    sleep 5
done

echo "============================================"
echo "All jobs completed at $(date)"
echo "============================================"