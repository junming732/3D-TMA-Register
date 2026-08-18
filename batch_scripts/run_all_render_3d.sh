#!/bin/bash

# =============================================================================
# run_all_render_3d.sh
# Step 6 (optional / on-demand QC): Interactive 3D mesh rendering of selected cells
#
# Requires Step 5's predecessor, link_3d_cells.py (Step 4), to have already run --
# this reads the 3D label volume + stats CSV it writes, and does not touch
# link_3d_cells.py or phenotype_cells.py itself.
# Writes one self-contained interactive HTML per rendered cell to
# ${INPUT_DIR_NAME}/<CORE_NAME>/qc/render_3d_qc/
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CORE_NUMS=(16 19)   # explicit list of core numbers to run

# Rendering flags -- leave CELL_IDS empty to render the same default random
# sample render_3d_cells.py draws (same seed/min_confirmed/n_samples as
# link_3d_cells.py's 2D tile QC montages). Set CELL_IDS to render specific
# cells instead, e.g. CELL_IDS="105412,88213" or CELL_IDS="1-50".
CELL_IDS=""
MIN_CONFIRMED=2
N_SAMPLES=50

SHOW_NEIGHBORS=true       # set to false to disable surrounding-cell context
CONTEXT_PAD_VOXELS=20
MAX_NEIGHBORS=25

# Registration-variant-dependent folder -- must match whatever produced your
# current 3D label volume / stats CSV. Change this (and nothing else) to
# point at a different registration run without editing render_3d_cells.py.
INPUT_DIR_NAME="CellPose_DAPI_3D_Valis"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RENDER_SCRIPT="${PROJECT_ROOT}/spatial_analysis/render_3d_cells.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_RENDER="${LOG_ROOT}/render_3d_valis"

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

mkdir -p "${LOG_RENDER}"

TOTAL=${#CORE_NUMS[@]}
DONE=0
FAIL=0
SKIP=0

declare -A CORE_STATUS

# Build optional flag strings
CELL_IDS_FLAG=""
if [ -n "${CELL_IDS}" ]; then
    CELL_IDS_FLAG="--cell_ids ${CELL_IDS}"
fi

NEIGHBORS_FLAG=""
if [ "${SHOW_NEIGHBORS}" = false ]; then
    NEIGHBORS_FLAG="--no_neighbors"
fi

echo "============================================================"
echo "  3D Cell Mesh Rendering (QC)"
echo "  Cores     : $(printf "Core_%02d " "${CORE_NUMS[@]}")"
if [ -n "${CELL_IDS}" ]; then
    echo "  Cell IDs  : ${CELL_IDS}"
else
    echo "  Cell IDs  : default sample (seed=0, min_confirmed=${MIN_CONFIRMED}, n=${N_SAMPLES})"
fi
echo "  Neighbors : ${SHOW_NEIGHBORS}"
echo "  Start time: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
IDX=0
for i in "${CORE_NUMS[@]}"; do

    CORE_NAME="Core_$(printf "%02d" $i)"
    IDX=$((IDX + 1))

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')]  ${CORE_NAME}  (${IDX}/${TOTAL})"
    echo "------------------------------------------------------------"

    # Skip if the 3D label volume / stats from link_3d_cells.py don't exist
    CORE_3D_DIR="${DATASPACE}${INPUT_DIR_NAME}/${CORE_NAME}"
    LABEL_TIF="${CORE_3D_DIR}/${CORE_NAME}_DAPI_3d_labels.tif"
    STATS_CSV="${CORE_3D_DIR}/${CORE_NAME}_DAPI_3d_stats.csv"

    if [ ! -f "${LABEL_TIF}" ] || [ ! -f "${STATS_CSV}" ]; then
        echo "  [SKIP] Missing 3D label volume and/or stats CSV in ${CORE_3D_DIR}"
        echo "         Run link_3d_cells.py (3D linkage step) first."
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_3D_OUTPUT"
        continue
    fi

    echo "  [RUN] Rendering 3D meshes..."

    python "${RENDER_SCRIPT}" \
        --core_name "${CORE_NAME}" \
        --min_confirmed ${MIN_CONFIRMED} \
        --n_samples ${N_SAMPLES} \
        --input_dir_name "${INPUT_DIR_NAME}" \
        --context_pad_voxels ${CONTEXT_PAD_VOXELS} \
        --max_neighbors ${MAX_NEIGHBORS} \
        ${CELL_IDS_FLAG} \
        ${NEIGHBORS_FLAG} \
        > "${LOG_RENDER}/${CORE_NAME}.log" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        FAIL=$((FAIL + 1))
        CORE_STATUS[$CORE_NAME]="FAIL"
        echo "  [FAIL] Rendering failed."
        echo "         Log: ${LOG_RENDER}/${CORE_NAME}.log"
        echo "         --- last 10 lines ---"
        tail -n 10 "${LOG_RENDER}/${CORE_NAME}.log" | sed 's/^/         /'
        echo "         ---------------------"
    else
        DONE=$((DONE + 1))
        CORE_STATUS[$CORE_NAME]="OK"
        # Extract rendered/skipped counts from log for inline reporting
        SUMMARY_LINE=$(grep "Done.  Rendered:" "${LOG_RENDER}/${CORE_NAME}.log" | tail -1)
        echo "  [OK]   ${SUMMARY_LINE:-Rendering complete.}"
    fi

done

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  3D rendering complete -- $(date)"
echo "------------------------------------------------------------"
echo "  Cores processed : ${TOTAL}"
printf "  Results         : %d OK  |  %d FAILED  |  %d SKIPPED  (of %d)\n" \
       $DONE $FAIL $SKIP $TOTAL
echo "------------------------------------------------------------"

echo "  Per-core status:"
for i in "${CORE_NUMS[@]}"; do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs  : ${LOG_RENDER}/"
echo "  Output: ${DATASPACE}${INPUT_DIR_NAME}/<CORE_NAME>/qc/render_3d_qc/cell_<ID>_3d.html"
echo "============================================================"