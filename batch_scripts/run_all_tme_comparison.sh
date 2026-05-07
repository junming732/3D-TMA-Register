#!/bin/bash

# =============================================================================
# run_all_tme_comparison.sh
# Step 7: 2D vs 3D TME spatial analysis
#
# Requires Steps 5-6 (phenotyping + cell-type assignment) to have completed.
# Reads from Phenotypes/<CORE>/<CORE>_phenotypes_typed.csv
#           Phenotypes/<CORE>/<CORE>_3d_typed.csv
# Writes  to TME_Analysis/<CORE>/
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Neighbourhood radius for spatial interaction scoring (µm)
RADIUS_UM=50

# Number of permutations for interaction score null distribution.
# Use 100 for a quick first-pass check, 1000 for final results.
N_PERM=1000

# Minimum number of cells of a type required to include it in analysis.
# Types below this threshold are skipped for that core.
MIN_CELLS=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ANALYSIS_SCRIPT="${PROJECT_ROOT}/registration/compare_2d_3d_tme.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_TME="${LOG_ROOT}/tme_comparison"

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

if [ ! -f "${ANALYSIS_SCRIPT}" ]; then
    echo "[ERROR] compare_2d_3d_tme.py not found at: ${ANALYSIS_SCRIPT}"
    exit 1
fi

mkdir -p "${LOG_TME}"

TOTAL=$((END - START + 1))
DONE=0
FAIL=0
SKIP=0

declare -A CORE_STATUS

echo "============================================================"
echo "  2D vs 3D TME Comparison Pipeline"
echo "  Cores          : Core_$(printf "%02d" $START) -> Core_$(printf "%02d" $END)"
echo "  Radius         : ${RADIUS_UM} um"
echo "  Permutations   : ${N_PERM}"
echo "  Min cells/type : ${MIN_CELLS}"
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
    # PREREQUISITE CHECK 1: typed 2D phenotype CSV
    # ------------------------------------------------------------------
    TYPED_2D="${DATASPACE}Phenotypes/${CORE_NAME}/${CORE_NAME}_phenotypes_typed.csv"
    if [ ! -f "${TYPED_2D}" ]; then
        echo "  [SKIP] Typed 2D phenotype CSV not found -- run assign_phenotypes.py first."
        echo "         Expected: ${TYPED_2D}"
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_2D"
        continue
    fi

    # ------------------------------------------------------------------
    # PREREQUISITE CHECK 2: typed 3D cell catalogue
    # ------------------------------------------------------------------
    TYPED_3D="${DATASPACE}Phenotypes/${CORE_NAME}/${CORE_NAME}_3d_typed.csv"
    if [ ! -f "${TYPED_3D}" ]; then
        echo "  [SKIP] Typed 3D cell catalogue not found -- run assign_phenotypes.py first."
        echo "         Expected: ${TYPED_3D}"
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_3D"
        continue
    fi

    echo "  [RUN] TME spatial analysis for ${CORE_NAME} ..."
    echo "        (n_perm=${N_PERM} -- this may take 20-40 min per core)"

    python "${ANALYSIS_SCRIPT}" \
        --core_name  "${CORE_NAME}" \
        --radius_um  "${RADIUS_UM}" \
        --n_perm     "${N_PERM}"    \
        --min_cells  "${MIN_CELLS}" \
        > "${LOG_TME}/${CORE_NAME}.log" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        FAIL=$((FAIL + 1))
        CORE_STATUS[$CORE_NAME]="FAIL"
        echo "  [FAIL] compare_2d_3d_tme.py failed for ${CORE_NAME}."
        echo "         Log: ${LOG_TME}/${CORE_NAME}.log"
        echo "         --- last 10 lines ---"
        tail -n 10 "${LOG_TME}/${CORE_NAME}.log" | sed 's/^/         /'
        echo "         ---------------------"
    else
        DONE=$((DONE + 1))
        CORE_STATUS[$CORE_NAME]="OK"
        echo "  [OK]   Analysis complete."
        # Pull summary line counts from log
        grep "type-pairs\|Summary table" "${LOG_TME}/${CORE_NAME}.log" \
            | tail -2 | sed 's/^/         /'
    fi

done

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  TME comparison complete -- $(date)"
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
echo "  Logs   : ${LOG_TME}/"
echo "  Output : ${DATASPACE}TME_Analysis/<CORE>/"
echo "============================================================"