#!/bin/bash

# =============================================================================
# run_missing_landmark_accuracy.sh
#
# Tailored wrapper around run_all_landmark_accuracy.sh that runs ONLY the
# (core, pipeline) combinations that are NOT yet done, given:
#
#   Core_09  -- already run on ALL pipelines           -> skip entirely
#   Core_16  -- not run on any pipeline yet             -> run all 5
#   Core_19  -- already run on valis only               -> run the 4 roma variants
#
# Registered pipelines (5 total):
#   valis
#   roma_dapi_clahe
#   roma_3chfusion
#   roma_colour_lut
#   roma_ck_clahe
#
# Each "roma_*" variant is the same underlying --pipeline roma evaluation,
# just pointed at a different ROMA_DIR_NAME (output folder). We select the
# variant by exporting ROMA_DIR_NAME before calling run_all_landmark_accuracy.sh
# (which now falls back to it via ${ROMA_DIR_NAME:-default} -- see patched
# base script).
#
# Edit the JOBS table below whenever your annotated-core / pipeline coverage
# changes -- that's the only thing you should need to touch.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_all_landmark_accuracy.sh"

# Overlay channels to request for every run (matches what you were already
# passing by hand). Change here once if you want a different set everywhere.
OVERLAY_CHANNELS="dapi,ck,ck_clahe,color_lut"

# Confirmed output-folder names for each roma variant.
declare -A ROMA_DIRS=(
    [roma_dapi_clahe]="Filter_AKAZE_RoMaV2_Linear_Warp_map_multi_channel_dapi_clahe"
    [roma_3chfusion]="Filter_AKAZE_RoMaV2_Linear_Warp_map_multi_channel_3ch_fusion"
    [roma_colour_lut]="Filter_AKAZE_RoMaV2_Linear_Warp_map_multi_channel_color_lut"
    [roma_ck_clahe]="Filter_AKAZE_RoMaV2_Linear_Warp_map_multi_channel_ck_clahe"
)

# -----------------------------------------------------------------------------
# JOBS: "core:pipeline_key" pairs still outstanding.
# pipeline_key is either "valis" or one of the keys in ROMA_DIRS above.
# -----------------------------------------------------------------------------


JOBS=(
    
    "19:roma_ck_clahe"

    # Core_09 -- already done on everything, intentionally omitted.
)

echo "============================================================"
echo "  Running ${#JOBS[@]} missing (core, pipeline) jobs"
echo "============================================================"

n=0
for job in "${JOBS[@]}"; do
    core="${job%%:*}"
    pkey="${job#*:}"
    n=$((n + 1))

    echo ""
    echo ">>> [$n/${#JOBS[@]}] Core_$(printf "%02d" "$core")  ->  ${pkey}"

    if [ "$pkey" = "valis" ]; then
        "${BASE_SCRIPT}" --pipeline valis \
            --overlay_channels "${OVERLAY_CHANNELS}" \
            --start "${core}" --end "${core}"
    else
        dir_name="${ROMA_DIRS[$pkey]:-}"
        if [ -z "$dir_name" ]; then
            echo "[ERROR] Unknown pipeline key: ${pkey}"
            exit 1
        fi
        ROMA_DIR_NAME="${dir_name}" "${BASE_SCRIPT}" --pipeline roma \
            --overlay_channels "${OVERLAY_CHANNELS}" \
            --start "${core}" --end "${core}"
    fi
done

echo ""
echo "============================================================"
echo "  All missing jobs complete."
echo "============================================================"