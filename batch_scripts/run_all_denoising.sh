#!/bin/bash

# =============================================================================
# run_all_denoising.sh
# Step 4b: Dust-aware top-hat denoising across all cores
#
# Run this AFTER registration (Step 4) and BEFORE phenotyping (Step 5).
# Reads registered volumes from Filter_AKAZE_RoMaV2_Linear_Warp_map/<CORE>/
# Writes denoised OME-TIFFs to Denoised/<CORE>/<CORE>_denoised.ome.tif
#
# phenotype_cells.py auto-detects the denoised volume; if it is present it
# uses it, otherwise it falls back to the raw volume.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Denoising parameters (must match what phenotype_cells.py expects)
NUCLEUS_UM=5.0      # nucleus radius in µm — sets the SE size
PIXEL_UM=0.4961     # pixel size in µm
DUST_PCT=99         # percentile for dust-blob detection
WORKERS=4           # parallel threads per slice (raise if you have cores to spare)
OVERWRITE=true     # set to true to re-denoise cores that already have output
PLOT_QC=true        # set to false to skip QC plots and save time

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DENOISE_SCRIPT="${PROJECT_ROOT}/registration/denoise_volume.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_DENOISE="${LOG_ROOT}/denoising"

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

mkdir -p "${LOG_DENOISE}"

TOTAL=$((END - START + 1))
DONE=0
FAIL=0
SKIP=0
ALREADY=0

declare -A CORE_STATUS

# Build flags
OVERWRITE_FLAG=""
if [ "${OVERWRITE}" = true ]; then
    OVERWRITE_FLAG="--overwrite"
fi

QC_FLAG=""
if [ "${PLOT_QC}" = true ]; then
    QC_FLAG="--plot_qc"
fi

echo "============================================================"
echo "  Top-Hat Denoising Pipeline"
echo "  Cores     : Core_$(printf "%02d" $START) -> Core_$(printf "%02d" $END)"
echo "  Nucleus   : ${NUCLEUS_UM} µm  (SE radius)"
echo "  Pixel     : ${PIXEL_UM} µm"
echo "  Dust pct  : ${DUST_PCT}th percentile"
echo "  Workers   : ${WORKERS} threads/slice"
echo "  Overwrite : ${OVERWRITE}"
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

    # Skip if registered volume doesn't exist
    REG_VOL="${DATASPACE}Filter_AKAZE_RoMaV2_Linear_Warp_map/${CORE_NAME}/${CORE_NAME}_AKAZE_RoMaV2_Linear_Aligned.ome.tif"
    if [ ! -f "${REG_VOL}" ]; then
        echo "  [SKIP] Registered volume not found at ${REG_VOL}"
        echo "         Run registration first."
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_VOLUME"
        continue
    fi

    # Report if output already exists and overwrite is off (denoise_volume.py
    # will exit 0 cleanly in this case — we just surface it here for the log)
    DENOISED_OUT="${DATASPACE}Denoised/${CORE_NAME}/${CORE_NAME}_denoised.ome.tif"
    if [ -f "${DENOISED_OUT}" ] && [ "${OVERWRITE}" = false ]; then
        echo "  [SKIP] Denoised volume already exists -- skipping."
        echo "         Set OVERWRITE=true to re-run."
        ALREADY=$((ALREADY + 1))
        CORE_STATUS[$CORE_NAME]="ALREADY_DONE"
        continue
    fi

    echo "  [RUN] Denoising all channels..."

    python "${DENOISE_SCRIPT}" \
        --core_name  "${CORE_NAME}" \
        --pixel_um   "${PIXEL_UM}" \
        --dust_pct   "${DUST_PCT}" \
        --workers    "${WORKERS}" \
        ${QC_FLAG} \
        ${OVERWRITE_FLAG} \
        > "${LOG_DENOISE}/${CORE_NAME}.log" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        FAIL=$((FAIL + 1))
        CORE_STATUS[$CORE_NAME]="FAIL"
        echo "  [FAIL] Denoising failed."
        echo "         Log: ${LOG_DENOISE}/${CORE_NAME}.log"
        echo "         --- last 10 lines ---"
        tail -n 10 "${LOG_DENOISE}/${CORE_NAME}.log" | sed 's/^/         /'
        echo "         ---------------------"
    else
        DONE=$((DONE + 1))
        CORE_STATUS[$CORE_NAME]="OK"
        # Pull timing and file size from the log for inline reporting
        ELAPSED=$(grep -oP "All slices done in \K[0-9.]+" "${LOG_DENOISE}/${CORE_NAME}.log" | tail -1)
        FILE_GB=$(grep -oP "\(\K[0-9.]+ GB" "${LOG_DENOISE}/${CORE_NAME}.log" | tail -1)
        echo "  [OK]   Denoising complete -- ${ELAPSED:-?}s total  |  ${FILE_GB:-? GB} written."
    fi

done

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Denoising pipeline complete -- $(date)"
echo "------------------------------------------------------------"
echo "  Cores in range   : ${TOTAL}"
printf "  Results          : %d OK  |  %d FAILED  |  %d SKIPPED  |  %d ALREADY DONE\n" \
       $DONE $FAIL $SKIP $ALREADY
echo "------------------------------------------------------------"

echo "  Per-core status:"
for i in $(seq $START $END); do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs  : ${LOG_DENOISE}/"
echo "  Output: ${DATASPACE}Denoised/<CORE_NAME>/<CORE_NAME>_denoised.ome.tif"
echo "============================================================"