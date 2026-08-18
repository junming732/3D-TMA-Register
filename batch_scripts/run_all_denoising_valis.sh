#!/bin/bash

# =============================================================================
# run_all_denoising_valis.sh
# Step 4b (VALIS variant): Dust-aware top-hat denoising across all cores
#
# Identical to run_all_denoising.sh -- denoise_volume.py is registration-
# variant-agnostic (it just needs a ZCYX registered volume at
# <INPUT_DIR_NAME>/<CORE>/<CORE><INPUT_FILE_SUFFIX>), so this is the same
# script with INPUT_DIR_NAME / INPUT_FILE_SUFFIX / OUTPUT_DIR_NAME pointed
# at VALIS's output instead of AKAZE-RomaV2/BSpline's. No changes to
# denoise_volume.py itself.
#
# Run this AFTER run_all_valis_registration.sh and BEFORE phenotyping.
# Reads registered volumes from ${INPUT_DIR_NAME}/<CORE>/ (see CONFIGURATION
# below) and writes denoised OME-TIFFs to ${OUTPUT_DIR_NAME}/<CORE>/<CORE>_denoised.ome.tif
#
# phenotype_cells.py auto-detects the denoised volume; if it is present it
# uses it, otherwise it falls back to the raw volume.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# START=1
# END=30
CORE_NUMS=(9 8 1 10)   # <-- explicit list of core numbers to run (missing cores)

# Denoising parameters (must match what phenotype_cells.py expects)
NUCLEUS_UM=5.0      # nucleus radius in µm — sets the SE size
PIXEL_UM=0.4961     # pixel size in µm
DUST_PCT=99         # percentile for dust-blob detection
WORKERS=4           # parallel threads per slice (raise if you have cores to spare)
OVERWRITE=true     # set to true to re-denoise cores that already have output
PLOT_QC=true        # set to false to skip QC plots and save time

# Registration variant to read from / denoised output folder to write to.
# Change these three to point at a different registration variant instead of
# editing denoise_volume.py directly — must match its --input_dir_name,
# --input_file_suffix, --output_dir_name defaults/flags.
#
# VALIS variant: valis_register_core2.py writes its merged stack to
#   ${DATASPACE}VALIS_Filter_Eval/<CORE>/<CORE>_VALIS_baseline.ome.tiff
# NOTE the double "ff" -- VALIS's own writer only speaks .ome.tiff and
# normalizes to it regardless of what extension you pass merged_path,
# even though valis_register_core2.py's merged_path string is built with
# a single "f" (_VALIS_baseline.ome.tif). Confirmed against actual output
# on disk; suffix below matches the real file, not the source string.
INPUT_DIR_NAME="VALIS_Filter_Eval"
INPUT_FILE_SUFFIX="_VALIS_baseline.ome.tiff"
OUTPUT_DIR_NAME="Denoised_valis"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DENOISE_SCRIPT="${PROJECT_ROOT}/spatial_analysis/denoise_volume.py"

LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_DENOISE="${LOG_ROOT}/denoising_valis"

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

mkdir -p "${LOG_DENOISE}"

# TOTAL=$((END - START + 1))
TOTAL=${#CORE_NUMS[@]}
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
# echo "  Cores     : Core_$(printf "%02d" $START) -> Core_$(printf "%02d" $END)"
echo "  Cores     : $(printf "Core_%02d " "${CORE_NUMS[@]}")"
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
# for i in $(seq $START $END); do
IDX=0
for i in "${CORE_NUMS[@]}"; do

    CORE_NAME="Core_$(printf "%02d" $i)"
    # IDX=$((i - START + 1))
    IDX=$((IDX + 1))

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')]  ${CORE_NAME}  (${IDX}/${TOTAL})"
    echo "------------------------------------------------------------"

    # Skip if registered volume doesn't exist
    REG_VOL="${DATASPACE}${INPUT_DIR_NAME}/${CORE_NAME}/${CORE_NAME}${INPUT_FILE_SUFFIX}"
    if [ ! -f "${REG_VOL}" ]; then
        echo "  [SKIP] Registered volume not found at ${REG_VOL}"
        echo "         Run registration first."
        SKIP=$((SKIP + 1))
        CORE_STATUS[$CORE_NAME]="SKIP_NO_VOLUME"
        continue
    fi

    # Report if output already exists and overwrite is off (denoise_volume.py
    # will exit 0 cleanly in this case — we just surface it here for the log)
    DENOISED_OUT="${DATASPACE}${OUTPUT_DIR_NAME}/${CORE_NAME}/${CORE_NAME}_denoised.ome.tif"
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
        --input_dir_name    "${INPUT_DIR_NAME}" \
        --input_file_suffix "${INPUT_FILE_SUFFIX}" \
        --output_dir_name   "${OUTPUT_DIR_NAME}" \
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
# for i in $(seq $START $END); do
for i in "${CORE_NUMS[@]}"; do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs  : ${LOG_DENOISE}/"
echo "  Output: ${DATASPACE}${OUTPUT_DIR_NAME}/<CORE_NAME>/<CORE_NAME>_denoised.ome.tif"
echo "============================================================"