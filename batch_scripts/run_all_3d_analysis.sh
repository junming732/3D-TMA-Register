#!/bin/bash

# =============================================================================
# run_all_3d_analysis.sh
# Step 4: 3D connected-component analysis across registered CellPose masks
#
# Requires Steps 1-3 (registration → CellPose → warp) to have already run.
# Reads warped masks from CellPose_<CHANNEL>_Warped/<CORE_NAME>/
# Writes 3D label volumes + stats to CellPose_<CHANNEL>_3D/<CORE_NAME>/
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Channels to analyse (must match what was segmented and warped)
CELLPOSE_CHANNELS="CD3 CD31"

# 3D analysis flags — tune MAX_SLICES per cell type:
#   T-cells (CD3)  ~10 µm → 2-3 slices at 4.5 µm/section → MAX_SLICES=4
#   Vessels (CD31) larger  → MAX_SLICES=6
MIN_SLICES=1
MAX_SLICES=4
MIN_VOLUME=10
# Co-localisation: second channel to compare against, or leave empty to skip
# e.g. when running CD3, set COLOC_CHANNEL=CD163 to get nearest-neighbour distances
COLOC_CHANNEL="CD163"
COLOC_RADIUS_UM=50
MIN_CONFIRMED=2

ANALYSIS_FLAGS="--plot_qc --min_slices ${MIN_SLICES} --max_slices ${MAX_SLICES} --min_area_px ${MIN_VOLUME} --min_overlap 3 --min_confirmed ${MIN_CONFIRMED} --coloc_radius_um ${COLOC_RADIUS_UM}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ANALYSIS_SCRIPT="${PROJECT_ROOT}/registration/analyse_3d_cells.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_3D="${LOG_ROOT}/3d_analysis"

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
source "${VENV_PATH}/bin/activate"

DATASPACE="$(python -c "import sys; sys.path.insert(0,'${PROJECT_ROOT}'); import config; print(config.DATASPACE)")"
if [ -z "${DATASPACE}" ]; then
    echo "[ERROR] Could not read DATASPACE from config.py — aborting."
    exit 1
fi
echo "  DATASPACE : ${DATASPACE}"

mkdir -p "${LOG_3D}"

TOTAL=$((END - START + 1))
DONE=0
FAIL=0
SKIP=0

declare -A CORE_STATUS

echo "============================================================"
echo "  3D Cell Analysis Pipeline"
echo "  Cores     : Core_$(printf "%02d" $START) -> Core_$(printf "%02d" $END)"
echo "  Channels  : ${CELLPOSE_CHANNELS}"
echo "  Z-span    : ${MIN_SLICES} - ${MAX_SLICES} slices"
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

    CORE_ALL_OK=1

    for CH in $CELLPOSE_CHANNELS; do

        WARPED_DIR="${DATASPACE}CellPose_${CH}_Warped/${CORE_NAME}"

        # Skip if warped masks don't exist yet
        if [ ! -d "${WARPED_DIR}" ]; then
            echo "  [SKIP] ${CH}: no warped mask directory at ${WARPED_DIR}"
            echo "         Run registration + CellPose + warp first."
            SKIP=$((SKIP + 1))
            CORE_ALL_OK=0
            continue
        fi

        # Skip if no warped mask files inside
        N_MASKS=$(find "${WARPED_DIR}" -name "*${CH}*_warped.tif" | wc -l)
        if [ "${N_MASKS}" -eq 0 ]; then
            echo "  [SKIP] ${CH}: directory exists but contains no *_warped.tif files."
            SKIP=$((SKIP + 1))
            CORE_ALL_OK=0
            continue
        fi

        echo "  [1/1] 3D analysis -- channel ${CH}  (${N_MASKS} warped slices)..."
        # Pass coloc channel only if set and different from current channel
        COLOC_FLAG=""
        if [ -n "${COLOC_CHANNEL}" ] && [ "${COLOC_CHANNEL}" != "${CH}" ]; then
            COLOC_FLAG="--coloc_channel ${COLOC_CHANNEL}"
        fi
        python "${ANALYSIS_SCRIPT}" \
            --core_name "${CORE_NAME}" \
            --channel   "${CH}" \
            ${ANALYSIS_FLAGS} \
            ${COLOC_FLAG} \
            > "${LOG_3D}/${CORE_NAME}_${CH}.log" 2>&1
        EXIT_CODE=$?

        if [ $EXIT_CODE -ne 0 ]; then
            FAIL=$((FAIL + 1))
            CORE_ALL_OK=0
            echo "  [FAIL] 3D analysis (${CH}) failed."
            echo "         Log: ${LOG_3D}/${CORE_NAME}_${CH}.log"
            echo "         --- last 10 lines ---"
            tail -n 10 "${LOG_3D}/${CORE_NAME}_${CH}.log" | sed 's/^/         /'
            echo "         ---------------------"
        else
            DONE=$((DONE + 1))
            # Extract cell count from log for quick inline summary
            N_CELLS=$(grep "Final 3D cell count" "${LOG_3D}/${CORE_NAME}_${CH}.log" \
                      | tail -1 | grep -o '[0-9]*$')
            echo "  [OK]   3D analysis (${CH}) complete -- ${N_CELLS:-?} 3D cells."
        fi

    done

    if [ $CORE_ALL_OK -eq 1 ]; then
        CORE_STATUS[$CORE_NAME]="OK"
    else
        CORE_STATUS[$CORE_NAME]="FAIL_OR_SKIP"
    fi

done

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
N_CHANNELS=$(echo $CELLPOSE_CHANNELS | wc -w)
MAX_RUNS=$((TOTAL * N_CHANNELS))

echo ""
echo "============================================================"
echo "  3D analysis complete -- $(date)"
echo "------------------------------------------------------------"
echo "  Cores processed : ${TOTAL}"
printf "  Channel-runs    : %d OK  |  %d FAILED  |  %d SKIPPED  (of %d)\n" \
       $DONE $FAIL $SKIP $MAX_RUNS
echo "------------------------------------------------------------"

echo "  Per-core status:"
for i in $(seq $START $END); do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs  : ${LOG_3D}/"
echo "  Output: ${DATASPACE}CellPose_<CHANNEL>_3D/<CORE_NAME>/"
echo "============================================================"