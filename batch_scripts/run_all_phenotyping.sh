#!/bin/bash

# =============================================================================
# run_all_phenotyping.sh
# Step 5: Cell phenotyping across all cores
#
# Requires Steps 1-4 (registration -> CellPose DAPI -> warp) to have run.
# Reads warped DAPI masks from CellPose_DAPI_Warped/<CORE_NAME>/
# Reads registered volume from Filter_AKAZE_RoMaV2_Linear_Warp_map/<CORE_NAME>/
# Writes per-core phenotype CSVs to Phenotypes/<CORE_NAME>/
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Phenotyping flags
MIN_AREA_PX=200     # minimum nucleus area in pixels (consistent with CellPose min_size)
PLOT_QC=true        # set to false to skip QC plots and save time

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PHENO_SCRIPT="${PROJECT_ROOT}/registration/phenotype_cells.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_PHENO="${LOG_ROOT}/phenotyping"

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
source "${VENV_PATH}/bin/activate"

DATASPACE="$(python -c "import sys; sys.path.insert(0,'${PROJECT_ROOT}'); import config; print(config.DATASPACE)")"
if [ -z "${DATASPACE}" ]; then
    echo "[ERROR] Could not read DATASPACE from config.py -- aborting."
    exit 1
fi
echo "  DATASPACE : ${DATASPACE}"

mkdir -p "${LOG_PHENO}"

TOTAL=$((END - START + 1))
DONE=0
FAIL=0
SKIP=0

declare -A CORE_STATUS

# Build QC flag string
QC_FLAG=""
if [ "${PLOT_QC}" = true ]; then
    QC_FLAG="--plot_qc"
fi

echo "============================================================"
echo "  Cell Phenotyping Pipeline"
echo "  Cores     : Core_$(printf "%02d" $START) -> Core_$(printf "%02d" $END)"
echo "  Min area  : ${MIN_AREA_PX} px"
echo "  QC plots  : ${PLOT_QC}"
echo "  Start time: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
for i in $(seq $START $END); do

    CORE_NAME="Core_$(printf "%02d" $i)"
    IDX=$((i - START + 1))

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')]  ${CORE_NAME}  (${IDX}/${TOTAL})"
    echo "------------------------------------------------------------"

    # Skip if warped DAPI masks don't exist
    DAPI_MASK_DIR="${DATASPACE}CellPose_DAPI_Warped/${CORE_NAME}"
    if [ ! -d "${DAPI_MASK_DIR}" ]; then
        echo "  [SKIP] No warped DAPI mask directory at ${DAPI_MASK_DIR}"
        echo "         Run CellPose (DAPI) + warp first."
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_MASKS"
        continue
    fi

    N_MASKS=$(find "${DAPI_MASK_DIR}" -name "*_DAPI_cp_masks_warped.tif" | wc -l)
    if [ "${N_MASKS}" -eq 0 ]; then
        echo "  [SKIP] DAPI mask directory exists but contains no *_DAPI_cp_masks_warped.tif files."
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_MASKS"
        continue
    fi

    # Skip if registered volume doesn't exist
    REG_VOL="${DATASPACE}Filter_AKAZE_RoMaV2_Linear_Warp_map/${CORE_NAME}/${CORE_NAME}_AKAZE_RoMaV2_Linear_Aligned.ome.tif"
    if [ ! -f "${REG_VOL}" ]; then
        echo "  [SKIP] Registered volume not found at ${REG_VOL}"
        echo "         Run registration first."
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_VOLUME"
        continue
    fi

    echo "  [RUN] Phenotyping (${N_MASKS} DAPI mask slices)..."

    python "${PHENO_SCRIPT}" \
        --core_name   "${CORE_NAME}" \
        --min_area_px "${MIN_AREA_PX}" \
        ${QC_FLAG} \
        > "${LOG_PHENO}/${CORE_NAME}.log" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        FAIL=$((FAIL + 1))
        CORE_STATUS[$CORE_NAME]="FAIL"
        echo "  [FAIL] Phenotyping failed."
        echo "         Log: ${LOG_PHENO}/${CORE_NAME}.log"
        echo "         --- last 10 lines ---"
        tail -n 10 "${LOG_PHENO}/${CORE_NAME}.log" | sed 's/^/         /'
        echo "         ---------------------"
    else
        DONE=$((DONE + 1))
        CORE_STATUS[$CORE_NAME]="OK"
        # Extract summary stats from log for inline reporting
        N_CELLS=$(grep "Total nuclei" "${LOG_PHENO}/${CORE_NAME}.log" \
                  | tail -1 | grep -o '[0-9]*$')
        echo "  [OK]   Phenotyping complete -- ${N_CELLS:-?} total nuclei."
        # Print per-marker positivity rates from the log
        grep -E "^\S.*positive" "${LOG_PHENO}/${CORE_NAME}.log" \
            | sed 's/^/         /' || true
    fi

done

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Phenotyping pipeline complete -- $(date)"
echo "------------------------------------------------------------"
echo "  Cores processed : ${TOTAL}"
printf "  Results         : %d OK  |  %d FAILED  |  %d SKIPPED  (of %d)\n" \
       $DONE $FAIL $SKIP $TOTAL
echo "------------------------------------------------------------"

echo "  Per-core status:"
for i in $(seq $START $END); do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs  : ${LOG_PHENO}/"
echo "  Output: ${DATASPACE}Phenotypes/<CORE_NAME>/<CORE_NAME>_phenotypes.csv"
echo "============================================================"