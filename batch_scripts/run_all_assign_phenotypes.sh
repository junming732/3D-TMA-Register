#!/bin/bash

# =============================================================================
# run_all_assign_phenotypes.sh
# Step 5: Cell type assignment via marker codebook + 3D consensus labelling
#
# Dependency chain (must have completed for each core before running this):
#   Step 3 — phenotype_cells.py
#             → Phenotypes/<CORE>/<CORE>_phenotypes.csv
#   Step 4 — analyse_3d_cells.py  (run on DAPI channel)
#             → CellPose_DAPI_3D/<CORE>/<CORE>_DAPI_2d_to_3d_map.csv
#             → CellPose_DAPI_3D/<CORE>/<CORE>_DAPI_3d_stats.csv
#
# Outputs (written to Phenotypes/<CORE>/):
#   <CORE>_phenotypes_typed.csv   — 2D nuclei with cell_type column
#   <CORE>_3d_typed.csv           — 3D cells with consensus cell_type + confidence
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Must match the channel used in run_all_3d_analysis.sh (DAPI = all-nucleus linking)
LINKING_CHANNEL="DAPI"

# Minimum fraction of 2D slice votes that must agree for a 3D consensus cell type.
# Cells below this threshold are labelled 'Ambiguous'.
MIN_CONFIDENCE=0.5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ASSIGN_SCRIPT="${PROJECT_ROOT}/registration/assign_phenotypes.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_ASSIGN="${LOG_ROOT}/assign_phenotypes"

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

if [ ! -f "${ASSIGN_SCRIPT}" ]; then
    echo "[ERROR] assign_phenotypes.py not found at: ${ASSIGN_SCRIPT}"
    exit 1
fi

mkdir -p "${LOG_ASSIGN}"

TOTAL=$((END - START + 1))
DONE=0
FAIL=0
SKIP=0

declare -A CORE_STATUS

echo "============================================================"
echo "  Cell Type Assignment Pipeline"
echo "  Cores          : Core_$(printf "%02d" $START) -> Core_$(printf "%02d" $END)"
echo "  Linking channel: ${LINKING_CHANNEL}"
echo "  Min confidence : ${MIN_CONFIDENCE}"
echo "  Start time     : $(date)"
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

    # ------------------------------------------------------------------
    # PREREQUISITE CHECK 1: phenotype CSV from phenotype_cells.py
    # ------------------------------------------------------------------
    PHENOTYPE_CSV="${DATASPACE}Phenotypes/${CORE_NAME}/${CORE_NAME}_phenotypes.csv"
    if [ ! -f "${PHENOTYPE_CSV}" ]; then
        echo "  [SKIP] Phenotype CSV not found -- run phenotype_cells.py first."
        echo "         Expected: ${PHENOTYPE_CSV}"
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_PHENOTYPE"
        continue
    fi

    # ------------------------------------------------------------------
    # PREREQUISITE CHECK 2: DAPI 2D→3D map from analyse_3d_cells.py
    # ------------------------------------------------------------------
    MAP_CSV="${DATASPACE}CellPose_${LINKING_CHANNEL}_3D/${CORE_NAME}/${CORE_NAME}_${LINKING_CHANNEL}_2d_to_3d_map.csv"
    if [ ! -f "${MAP_CSV}" ]; then
        echo "  [SKIP] 2D→3D map CSV not found -- run analyse_3d_cells.py (DAPI) first."
        echo "         Expected: ${MAP_CSV}"
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_3D_MAP"
        continue
    fi

    # ------------------------------------------------------------------
    # PREREQUISITE CHECK 3: DAPI 3D stats from analyse_3d_cells.py
    # ------------------------------------------------------------------
    STATS_CSV="${DATASPACE}CellPose_${LINKING_CHANNEL}_3D/${CORE_NAME}/${CORE_NAME}_${LINKING_CHANNEL}_3d_stats.csv"
    if [ ! -f "${STATS_CSV}" ]; then
        echo "  [SKIP] 3D stats CSV not found -- run analyse_3d_cells.py (DAPI) first."
        echo "         Expected: ${STATS_CSV}"
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_3D_STATS"
        continue
    fi

    echo "  [RUN] Assigning cell types for ${CORE_NAME} ..."

    python "${ASSIGN_SCRIPT}" \
        --core_name        "${CORE_NAME}" \
        --linking_channel  "${LINKING_CHANNEL}" \
        --min_confidence   "${MIN_CONFIDENCE}" \
        > "${LOG_ASSIGN}/${CORE_NAME}.log" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        FAIL=$((FAIL + 1))
        CORE_STATUS[$CORE_NAME]="FAIL"
        echo "  [FAIL] assign_phenotypes.py failed for ${CORE_NAME}."
        echo "         Log: ${LOG_ASSIGN}/${CORE_NAME}.log"
        echo "         --- last 10 lines ---"
        tail -n 10 "${LOG_ASSIGN}/${CORE_NAME}.log" | sed 's/^/         /'
        echo "         ---------------------"
    else
        DONE=$((DONE + 1))
        CORE_STATUS[$CORE_NAME]="OK"

        # Quick inline summary: pull cell type counts from log
        echo "  [OK]   Assignment complete."
        grep "^ \{2\}" "${LOG_ASSIGN}/${CORE_NAME}.log" \
            | grep -E "(Tumour|T_cell|Macrophage|Endothelial|Neural|Unknown|Ambiguous)" \
            | sed 's/^/         /'
    fi

done

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Cell type assignment complete -- $(date)"
echo "------------------------------------------------------------"
printf "  Results : %d OK  |  %d FAILED  |  %d SKIPPED  (of %d cores)\n" \
       $DONE $FAIL $SKIP $TOTAL
echo "------------------------------------------------------------"
echo "  Per-core status:"
for i in $(seq $START $END); do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done
echo "------------------------------------------------------------"
echo "  Logs   : ${LOG_ASSIGN}/"
echo "  Output : ${DATASPACE}Phenotypes/<CORE>/*_typed.csv"
echo "============================================================"