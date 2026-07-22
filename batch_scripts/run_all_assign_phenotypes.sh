#!/bin/bash

# =============================================================================
# run_all_assign_phenotypes.sh
# Step 5: Cell type assignment via marker codebook + 3D consensus labelling
#
# Dependency chain (must have completed for each core before running this):
#   phenotype_cells.py
#             → ${PHENOTYPE_DIR_NAME}/<CORE>/<CORE>_phenotypes.csv
#   link_3d_cells.py  (DAPI only)
#             → ${LINKING_DIR_NAME}/<CORE>/<CORE>_DAPI_2d_to_3d_map.csv
#             → ${LINKING_DIR_NAME}/<CORE>/<CORE>_DAPI_3d_stats.csv
#
# Outputs (written to ${OUTPUT_DIR_NAME}/<CORE>/):
#   <CORE>_phenotypes_typed.csv   — 2D nuclei with cell_type column
#   <CORE>_3d_typed.csv           — 3D cells with consensus cell_type + confidence
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
START=1
END=30

# Minimum fraction of 2D slice votes that must agree for a 3D consensus cell type.
# Cells below this threshold are labelled 'Ambiguous'.
MIN_CONFIDENCE=0.5

# Registration-variant-dependent folders — must match whatever produced your
# current inputs. Change these to point at a different registration run
# without editing assign_phenotypes.py.
PHENOTYPE_DIR_NAME="Phenotypes_Bspline"
LINKING_DIR_NAME="CellPose_DAPI_3D_Bspline"
OUTPUT_DIR_NAME="Phenotypes_Bspline"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ASSIGN_SCRIPT="${PROJECT_ROOT}/spatial_analysis/assign_phenotypes.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_ASSIGN="${LOG_ROOT}/assign_phenotypes_Bspline"

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
# Read VENV_PATH with system python3, BEFORE activating any venv
VENV_PATH="$(python3 -c "import sys; sys.path.insert(0,'${PROJECT_ROOT}'); import config; print(config.VENV_PATH)")"
if [ -z "${VENV_PATH}" ]; then
    echo "[ERROR] Could not read VENV_PATH from config.py -- aborting."
    exit 1
fi

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
    PHENOTYPE_CSV="${DATASPACE}${PHENOTYPE_DIR_NAME}/${CORE_NAME}/${CORE_NAME}_phenotypes.csv"
    if [ ! -f "${PHENOTYPE_CSV}" ]; then
        echo "  [SKIP] Phenotype CSV not found -- run phenotype_cells.py first."
        echo "         Expected: ${PHENOTYPE_CSV}"
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_PHENOTYPE"
        continue
    fi

    # ------------------------------------------------------------------
    # PREREQUISITE CHECK 2: DAPI 2D→3D map from link_3d_cells.py
    # ------------------------------------------------------------------
    MAP_CSV="${DATASPACE}${LINKING_DIR_NAME}/${CORE_NAME}/${CORE_NAME}_DAPI_2d_to_3d_map.csv"
    if [ ! -f "${MAP_CSV}" ]; then
        echo "  [SKIP] 2D→3D map CSV not found -- run link_3d_cells.py (DAPI) first."
        echo "         Expected: ${MAP_CSV}"
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_3D_MAP"
        continue
    fi

    # ------------------------------------------------------------------
    # PREREQUISITE CHECK 3: DAPI 3D stats from link_3d_cells.py
    # ------------------------------------------------------------------
    STATS_CSV="${DATASPACE}${LINKING_DIR_NAME}/${CORE_NAME}/${CORE_NAME}_DAPI_3d_stats.csv"
    if [ ! -f "${STATS_CSV}" ]; then
        echo "  [SKIP] 3D stats CSV not found -- run link_3d_cells.py (DAPI) first."
        echo "         Expected: ${STATS_CSV}"
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_3D_STATS"
        continue
    fi

    echo "  [RUN] Assigning cell types for ${CORE_NAME} ..."

    python "${ASSIGN_SCRIPT}" \
        --core_name          "${CORE_NAME}" \
        --min_confidence     "${MIN_CONFIDENCE}" \
        --phenotype_dir_name "${PHENOTYPE_DIR_NAME}" \
        --linking_dir_name   "${LINKING_DIR_NAME}" \
        --output_dir_name    "${OUTPUT_DIR_NAME}" \
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
echo "  Output : ${DATASPACE}${OUTPUT_DIR_NAME}/<CORE>/*_typed.csv"
echo "============================================================"