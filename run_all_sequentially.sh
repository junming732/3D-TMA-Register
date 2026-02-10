#!/bin/bash

# --- CONFIGURATION ---
# Range of cores to process (Core_01 to Core_30)
START=1
END=30

# YOUR ACTUAL ENVIRONMENT PATH
VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Activate environment
source "${VENV_PATH}/bin/activate"

echo "============================================"
echo "Starting Sequential Processing: Core_$(printf "%02d" $START) to Core_$(printf "%02d" $END)"
echo "Script:      VALIS_register_core.py"
echo "Environment: ${VENV_PATH}"
echo "Machine:     $(hostname)"
echo "Start Time:  $(date)"
echo "============================================"

# Loop through 1 to 30
for i in $(seq $START $END); do
    
    # Format the core name with underscore and 2 digits (Core_01, Core_02...)
    CORE_NAME="Core_$(printf "%02d" $i)"
    
    echo "--------------------------------------------"
    echo "[$(date)] Processing ${CORE_NAME}..."
    
    # Run the Python script (Updated filename)
    # Logs are saved to Core_01.log, Core_02.log etc.
    python VALIS_register_core.py --core_name "${CORE_NAME}" --channel_idx 0 > "${CORE_NAME}.log" 2>&1
    
    # Check status
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] ${CORE_NAME} finished."
    else
        echo "[FAILURE] ${CORE_NAME} failed. See ${CORE_NAME}.log"
    fi
    
    # Pause to let GPU cool down/memory flush
    sleep 5
done

echo "============================================"
echo "All jobs completed at $(date)"
echo "============================================"