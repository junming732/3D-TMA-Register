#!/bin/bash

# =============================================================================
# run_all_3d_analysis.sh
# Step 4: 3D connected-component analysis across registered CellPose masks
#
# Requires Steps 1-3 (registration -> CellPose -> warp) to have already run.
# Reads warped masks from CellPose_<CHANNEL>_Warped/<CORE_NAME>/
# Writes 3D label volumes + stats to CellPose_<CHANNEL>_3D/<CORE_NAME>/
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Channels to analyse. CD31 isolated for future testing.
CELLPOSE_CHANNELS="DAPI"

# =============================================================================
# ARGUMENT GUIDE — what each flag controls and why they do NOT conflict
# =============================================================================
# These filters act at DIFFERENT stages of the pipeline:
#
#  Stage 1 — LINK ACCEPTANCE (per adjacent-slice pair):
#    --min_overlap         : raw pixel count gate.  First rough filter.
#    --min_overlap_frac    : overlap / smaller_area. Normalises for size diff.
#    --min_iou             : overlap / union.  Stricter; penalises area mismatch.
#    --min_intensity_frac  : mean DAPI of each slice >= fraction of cell peak.
#                            Catches dark/empty mid-stack CellPose masks.
#    -> These work TOGETHER. A link must pass ALL active thresholds.
#       They are complementary, not contradictory.
#
#  Stage 2 — COMPONENT PRUNING (after graph is built):
#    --max_slices          : severs weakest edge if z_max - z_min + 1 > limit.
#                            Controls Z-SPAN (height of the 3D cell).
#    --max_segments_per_z  : rejects cells with >N masks on a single Z-slice.
#                            Catches CellPose over-segmentation (the "8-panel
#                            but span=3" bug shown in cell_105412).
#    -> These work on the component AFTER all links are accepted.
#
#  Stage 3 — OUTPUT FILTERING (keep / discard whole 3D cells):
#    --min_slices          : minimum z_span to keep a cell (removes singletons).
#    --min_area_px         : minimum peak 2D area.
#    --min_confirmed       : minimum z_span to count as "confirmed" in QC.
# =============================================================================

# Global 3D analysis flags
MIN_SLICES=1
MIN_VOLUME=10
MIN_CONFIRMED=2
MIN_IOU=0.15
MIN_INTENSITY_FRAC=0.3

# Co-localisation: second channel to compare against, or leave empty to skip
COLOC_CHANNEL=""
COLOC_RADIUS_UM=50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ANALYSIS_SCRIPT="${PROJECT_ROOT}/registration/link_3d_cells.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
# LOG_3D="${LOG_ROOT}/3d_analysis"
LOG_3D="${LOG_ROOT}/3d_linkage_bspline"

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

    CORE_ALL_OK=1

    for CH in $CELLPOSE_CHANNELS; do

        # --- DYNAMIC BIOLOGICAL CONSTRAINTS ---
        # Assign Z-span limits based on the specific morphology of the channel
        if [ "$CH" == "DAPI" ]; then
            MAX_SLICES=5   # Nuclei: ~4.5 um sections, nucleus ~10 um diameter → 2-4 slices max
        elif [ "$CH" == "CD3" ]; then
            MAX_SLICES=3   # T-cells: strict topological severing
        elif [ "$CH" == "CD31" ]; then
            MAX_SLICES=15  # Vessels: allow long continuous structures (Inactive)
        else
            MAX_SLICES=4   # Default fallback
        fi

        #WARPED_DIR="${DATASPACE}CellPose_${CH}_Warped/${CORE_NAME}"
        WARPED_DIR="${DATASPACE}CellPose_${CH}_Warped_Bspline/${CORE_NAME}"

        # Skip if warped masks do not exist yet
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

        # Per-channel tuning: nuclei are much larger than T-cells
        if [ "$CH" = "DAPI" ]; then
            CH_MIN_AREA=200   # consistent with CellPose min_size for DAPI
            CH_MIN_OVERLAP=30 # larger nuclei need more overlap to confirm a true link
        else
            CH_MIN_AREA=${MIN_VOLUME}
            CH_MIN_OVERLAP=3
        fi

        echo "  [RUN] 3D analysis -- channel ${CH}  (Max Slices: ${MAX_SLICES}) (${N_MASKS} warped slices)..."
        
        # Pass coloc channel only if set and different from current channel
        COLOC_FLAG=""
        if [ -n "${COLOC_CHANNEL}" ] && [ "${COLOC_CHANNEL}" != "${CH}" ]; then
            COLOC_FLAG="--coloc_channel ${COLOC_CHANNEL}"
        fi
        
        python "${ANALYSIS_SCRIPT}" \
            --core_name "${CORE_NAME}" \
            --channel   "${CH}" \
            --plot_qc \
            --min_slices ${MIN_SLICES} \
            --max_slices ${MAX_SLICES} \
            --min_area_px ${CH_MIN_AREA} \
            --min_overlap ${CH_MIN_OVERLAP} \
            --min_confirmed ${MIN_CONFIRMED} \
            --coloc_radius_um ${COLOC_RADIUS_UM} \
            --min_iou ${MIN_IOU} \
            --min_intensity_frac ${MIN_INTENSITY_FRAC} \
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

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
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
# echo "  Output: ${DATASPACE}CellPose_<CHANNEL>_3D/<CORE_NAME>/"
echo "============================================================"