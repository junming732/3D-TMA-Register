#!/bin/bash

# =============================================================================
# run_registration.sh
# Step 1 only: AKAZE affine + RoMaV2 dense warp (saves deformation maps)
#
# Run this first across all cores before running run_cellpose.sh.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REG_SCRIPT="${PROJECT_ROOT}/registration/akaze_linear_romav2_warp_map.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_REG="${LOG_ROOT}/registration"

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
source "${VENV_PATH}/bin/activate"

mkdir -p "${LOG_REG}"

TOTAL=$((END - START + 1))
DONE_REG=0
FAIL_REG=0

declare -A CORE_STATUS

echo "============================================================"
echo "  Registration Pipeline: AKAZE affine → RoMaV2 dense warp"
echo "  Cores     : Core_$(printf "%02d" $START) → Core_$(printf "%02d" $END)"
echo "  Start time: $(date)"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
for i in $(seq $START $END); do

    CORE_NAME="Core_$(printf "%02d" $i)"
    IDX=$((DONE_REG + FAIL_REG + 1))

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')]  ${CORE_NAME}  (${IDX}/${TOTAL})"
    echo "------------------------------------------------------------"

    echo "  [1/1] Registration..."
    python "${REG_SCRIPT}" --core_name "${CORE_NAME}" \
        > "${LOG_REG}/${CORE_NAME}.log" 2>&1
    REG_EXIT=$?

    if [ $REG_EXIT -ne 0 ]; then
        FAIL_REG=$((FAIL_REG + 1))
        CORE_STATUS[$CORE_NAME]="FAIL_REG"
        echo "  [FAIL] Registration failed."
        echo "         Log: ${LOG_REG}/${CORE_NAME}.log"
        echo "         --- last 10 lines ---"
        tail -n 10 "${LOG_REG}/${CORE_NAME}.log" | sed 's/^/         /'
        echo "         ---------------------"
    else
        DONE_REG=$((DONE_REG + 1))
        CORE_STATUS[$CORE_NAME]="OK"
        echo "  [OK]   Registration complete."
    fi

done

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Registration complete — $(date)"
echo "------------------------------------------------------------"
echo "  Cores processed : $((DONE_REG + FAIL_REG)) / ${TOTAL}"
printf "  Registration    : %d OK  |  %d FAILED\n" $DONE_REG $FAIL_REG
echo "------------------------------------------------------------"

echo "  Per-core status:"
for i in $(seq $START $END); do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs: ${LOG_REG}/"
echo "============================================================"