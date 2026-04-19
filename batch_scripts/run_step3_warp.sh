#!/bin/bash

# =============================================================================
# run_step3_warp.sh
# Step 3: CellPose mask warping
#
# Requires registration to have already been run (run_registration.sh)
# and CellPose segmentation to have been completed.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Warp script extra flags
WARP_FLAGS="--plot_qc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WARP_SCRIPT="${PROJECT_ROOT}/registration/warp_cellpose_masks.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_WARP="${LOG_ROOT}/warp"

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

mkdir -p "${LOG_WARP}"

TOTAL=$((END - START + 1))
DONE_WARP=0; FAIL_WARP=0

declare -A CORE_STATUS

echo "============================================================"
echo "  CellPose Pipeline: Mask Warp"
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

    # ── STEP 3: Warp masks ─────

    MASK_DIR="${DATASPACE}CellPose_DAPI/${CORE_NAME}"
    DEFORM_DIR="${DATASPACE}Filter_AKAZE_RoMaV2_Linear_Warp_map/${CORE_NAME}/deformation_maps"
    OUT_DIR="${DATASPACE}CellPose_DAPI_Warped/${CORE_NAME}"

    if [ ! -d "${MASK_DIR}" ]; then
        echo "  [SKIP] Warp: no mask directory found at ${MASK_DIR}"
        CORE_STATUS[$CORE_NAME]="SKIP_NO_MASK"
    elif [ ! -d "${DEFORM_DIR}" ]; then
        echo "  [SKIP] Warp: no deformation maps at ${DEFORM_DIR} — run registration first."
        CORE_STATUS[$CORE_NAME]="SKIP_NO_DEFORM"
    else
        echo "  [1/1] Warp masks — DAPI..."
        python "${WARP_SCRIPT}" \
            --core_name  "${CORE_NAME}" \
            --mask_dir   "${MASK_DIR}" \
            --deform_dir "${DEFORM_DIR}" \
            --out_dir    "${OUT_DIR}" \
            ${WARP_FLAGS} \
            > "${LOG_WARP}/${CORE_NAME}_DAPI.log" 2>&1
        WARP_EXIT=$?

        if [ $WARP_EXIT -ne 0 ]; then
            FAIL_WARP=$((FAIL_WARP + 1))
            CORE_STATUS[$CORE_NAME]="FAIL_WARP"
            echo "  [FAIL] Warp failed."
            echo "         Log: ${LOG_WARP}/${CORE_NAME}_DAPI.log"
            echo "         --- last 10 lines ---"
            tail -n 10 "${LOG_WARP}/${CORE_NAME}_DAPI.log" | sed 's/^/         /'
            echo "         ---------------------"
        else
            DONE_WARP=$((DONE_WARP + 1))
            CORE_STATUS[$CORE_NAME]="OK"
            echo "  [OK]   Warp complete."
        fi
    fi

done

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Warp pipeline complete — $(date)"
echo "------------------------------------------------------------"
echo "  Cores processed : ${TOTAL}"
printf "  Mask warp (DAPI): %d OK  |  %d FAILED\n" $DONE_WARP $FAIL_WARP
echo "------------------------------------------------------------"

echo "  Per-core status:"
for i in $(seq $START $END); do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs:"
echo "    Warp     : ${LOG_WARP}/"
echo "============================================================"