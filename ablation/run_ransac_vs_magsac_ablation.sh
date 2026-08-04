#!/bin/bash

# =============================================================================
# run_ransac_vs_magsac_ablation.sh
# Step 1 ablation: RANSAC vs USAC_MAGSAC, isolated (L0 only, no GPU needed).
#
# Mirrors run_all_akaze_roma_multi_channel_map.sh's venv/logging conventions.
# Unlike the production script, this makes ONE python call for the whole
# core range (the harness loops internally and isolates per-core failures),
# since there's no per-core GPU state to worry about.
# =============================================================================

START=1
END=30

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ABLATION_SCRIPT="${SCRIPT_DIR}/test_ransac_vs_magsac.py"

LOG_ROOT="${PROJECT_ROOT}/log/ablation"
LOG_FILE="${LOG_ROOT}/ransac_vs_magsac_Core_$(printf "%02d" $START)-$(printf "%02d" $END).log"

VENV_PATH="$(python3 -c "import sys; sys.path.insert(0,'${PROJECT_ROOT}'); import config; print(config.VENV_PATH)")"
if [ -z "${VENV_PATH}" ]; then
    echo "[ERROR] Could not read VENV_PATH from config.py -- aborting."
    exit 1
fi
source "${VENV_PATH}/bin/activate"

mkdir -p "${LOG_ROOT}"

echo "============================================================"
echo "  RANSAC vs USAC_MAGSAC ablation (L0 only)"
echo "  Cores     : Core_$(printf "%02d" $START) → Core_$(printf "%02d" $END)"
echo "  Start time: $(date)"
echo "  Log       : ${LOG_FILE}"
echo "============================================================"

python "${ABLATION_SCRIPT}" --start "${START}" --end "${END}" \
    2>&1 | tee "${LOG_FILE}"

echo "============================================================"
echo "  Ablation complete — $(date)"
echo "  CSV: ${SCRIPT_DIR}/ransac_vs_magsac_ablation.csv"
echo "============================================================"