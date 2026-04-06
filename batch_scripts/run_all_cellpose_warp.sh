#!/bin/bash

# =============================================================================
# run_cellpose.sh
# Steps 2 & 3: CellPose segmentation → CellPose mask warping
#
# Requires registration to have already been run (run_registration.sh).
# A core with no deformation maps is skipped at Step 3.
# A core that fails at Step 2 is skipped at Step 3.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Channels to segment with CellPose (space-separated; must match CHANNEL_CONFIGS
# in cellpose_segementation.py).  Add/remove as needed.
CELLPOSE_CHANNELS="CD3 CD31"

# CellPose extra flags (applied to every channel run).
GPU_FLAGS="--use_gpu"
CELLPOSE_FLAGS="--plot_qc"

# Warp script extra flags
WARP_FLAGS="--plot_qc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CP_SCRIPT="${PROJECT_ROOT}/registration/cellpose_segmentation.py"
WARP_SCRIPT="${PROJECT_ROOT}/registration/warp_cellpose_masks.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_CP="${LOG_ROOT}/cellpose"
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

mkdir -p "${LOG_CP}" "${LOG_WARP}"

TOTAL=$((END - START + 1))
DONE_CP=0;   FAIL_CP=0
DONE_WARP=0; FAIL_WARP=0

declare -A CORE_STATUS

echo "============================================================"
echo "  CellPose Pipeline: Segmentation → Mask Warp"
echo "  Cores     : Core_$(printf "%02d" $START) → Core_$(printf "%02d" $END)"
echo "  Channels  : ${CELLPOSE_CHANNELS}"
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

    # ── STEP 2: CellPose segmentation (all requested channels) ───────────────
    CP_ALL_OK=1

    for CH in $CELLPOSE_CHANNELS; do
        echo "  [1/2] CellPose — channel ${CH}..."
        python "${CP_SCRIPT}" \
            --core_name "${CORE_NAME}" \
            --channel   "${CH}" \
            ${CELLPOSE_FLAGS} \
            ${GPU_FLAGS} \
            > "${LOG_CP}/${CORE_NAME}_${CH}.log" 2>&1
        CP_EXIT=$?

        if [ $CP_EXIT -ne 0 ]; then
            CP_ALL_OK=0
            FAIL_CP=$((FAIL_CP + 1))
            echo "  [FAIL] CellPose (${CH}) failed — skipping warp for this channel."
            echo "         Log: ${LOG_CP}/${CORE_NAME}_${CH}.log"
            echo "         --- last 10 lines ---"
            tail -n 10 "${LOG_CP}/${CORE_NAME}_${CH}.log" | sed 's/^/         /'
            echo "         ---------------------"
        else
            DONE_CP=$((DONE_CP + 1))
            echo "  [OK]   CellPose (${CH}) complete."
        fi
    done

    if [ $CP_ALL_OK -eq 0 ]; then
        CORE_STATUS[$CORE_NAME]="FAIL_CP"
        # Still attempt warping for channels that did succeed (handled per-channel below)
    fi

    # ── STEP 3: Warp masks (per channel, only if CellPose output exists) ─────
    WARP_ALL_OK=1

    for CH in $CELLPOSE_CHANNELS; do

        MASK_DIR="${DATASPACE}CellPose_${CH}/${CORE_NAME}"
        DEFORM_DIR="${DATASPACE}Filter_AKAZE_RoMaV2_Linear_Warp_map/${CORE_NAME}/deformation_maps"
        OUT_DIR="${DATASPACE}CellPose_${CH}_Warped/${CORE_NAME}"

        # Skip if CellPose produced no output for this channel
        if [ ! -d "${MASK_DIR}" ]; then
            echo "  [SKIP] Warp (${CH}): no mask directory found at ${MASK_DIR}"
            continue
        fi

        # Skip if registration deformation maps are missing
        if [ ! -d "${DEFORM_DIR}" ]; then
            echo "  [SKIP] Warp (${CH}): no deformation maps at ${DEFORM_DIR} — run registration first."
            continue
        fi

        echo "  [2/2] Warp masks — channel ${CH}..."
        python "${WARP_SCRIPT}" \
            --core_name  "${CORE_NAME}" \
            --mask_dir   "${MASK_DIR}" \
            --deform_dir "${DEFORM_DIR}" \
            --out_dir    "${OUT_DIR}" \
            --channel    "${CH}" \
            ${WARP_FLAGS} \
            > "${LOG_WARP}/${CORE_NAME}_${CH}.log" 2>&1
        WARP_EXIT=$?

        if [ $WARP_EXIT -ne 0 ]; then
            WARP_ALL_OK=0
            FAIL_WARP=$((FAIL_WARP + 1))
            echo "  [FAIL] Warp (${CH}) failed."
            echo "         Log: ${LOG_WARP}/${CORE_NAME}_${CH}.log"
            echo "         --- last 10 lines ---"
            tail -n 10 "${LOG_WARP}/${CORE_NAME}_${CH}.log" | sed 's/^/         /'
            echo "         ---------------------"
        else
            DONE_WARP=$((DONE_WARP + 1))
            echo "  [OK]   Warp (${CH}) complete."
        fi
    done

    if [ $CP_ALL_OK -eq 1 ] && [ $WARP_ALL_OK -eq 1 ]; then
        CORE_STATUS[$CORE_NAME]="OK"
    elif [ -z "${CORE_STATUS[$CORE_NAME]}" ]; then
        CORE_STATUS[$CORE_NAME]="FAIL_WARP"
    fi

done

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
N_CHANNELS=$(echo $CELLPOSE_CHANNELS | wc -w)
MAX_CP=$((TOTAL * N_CHANNELS))
MAX_WARP=$MAX_CP

echo ""
echo "============================================================"
echo "  CellPose pipeline complete — $(date)"
echo "------------------------------------------------------------"
echo "  Cores processed : ${TOTAL}"
printf "  CellPose        : %d OK  |  %d FAILED  (of %d channel-runs)\n" \
       $DONE_CP $FAIL_CP $MAX_CP
printf "  Mask warp       : %d OK  |  %d FAILED  (of %d channel-runs)\n" \
       $DONE_WARP $FAIL_WARP $MAX_WARP
echo "------------------------------------------------------------"

echo "  Per-core status:"
for i in $(seq $START $END); do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs:"
echo "    CellPose : ${LOG_CP}/"
echo "    Warp     : ${LOG_WARP}/"
echo "============================================================"