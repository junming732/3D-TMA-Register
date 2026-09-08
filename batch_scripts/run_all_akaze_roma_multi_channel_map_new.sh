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

# Cores to skip entirely (not run, not counted as FAIL)
EXCLUDED_CORES=(9 16 19 17 21 23 27)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REG_SCRIPT="${PROJECT_ROOT}/registration/akaze_romav2_multi_channel_warp_new.py"

REGISTRATION_MODE="dapi_clahe"     # ck_only | 3ch_fusion | color_lut | dapi_clahe | ck_clahe
                                  # ck_clahe adopted per ablation (428/512 pairs improved,
                                  # p<0.0001, Friedman-confirmed). dapi_clahe/3ch_fusion/
                                  # color_lut were all significantly WORSE than ck_only.

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_REG="${LOG_ROOT}/registration/akaze_roma_${REGISTRATION_MODE}"

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

mkdir -p "${LOG_REG}"

is_excluded() {
    local core_num=$1
    for ex in "${EXCLUDED_CORES[@]}"; do
        if [ "$ex" -eq "$core_num" ]; then
            return 0
        fi
    done
    return 1
}

TOTAL=0
for i in $(seq $START $END); do
    if ! is_excluded "$i"; then
        TOTAL=$((TOTAL + 1))
    fi
done

DONE_REG=0
FAIL_REG=0
SKIP_REG=0

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

    if is_excluded "$i"; then
        SKIP_REG=$((SKIP_REG + 1))
        CORE_STATUS[$CORE_NAME]="SKIPPED"
        echo ""
        echo "------------------------------------------------------------"
        echo "[$(date '+%H:%M:%S')]  ${CORE_NAME}  -- SKIPPED (excluded)"
        echo "------------------------------------------------------------"
        continue
    fi

    IDX=$((DONE_REG + FAIL_REG + 1))

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')]  ${CORE_NAME}  (${IDX}/${TOTAL})"
    echo "------------------------------------------------------------"

    echo "  [1/1] Registration..."
    python "${REG_SCRIPT}" --core_name "${CORE_NAME}" \
    --registration_mode "${REGISTRATION_MODE}" \
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
echo "  Cores processed : $((DONE_REG + FAIL_REG)) / ${TOTAL}  (excluded: ${SKIP_REG})"
printf "  Registration    : %d OK  |  %d FAILED  |  %d SKIPPED\n" $DONE_REG $FAIL_REG $SKIP_REG
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