#!/bin/bash

# --- CONFIGURATION ---
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"
LOG_DIR="log/evaluation"
CHANNEL=6          # CK channel (change if needed)
BINS_MI=64         # MI histogram bins

# Output folder where per-core CSVs will land (mirrors config.DATASPACE)
# Aggregate summary will be written here
AGGREGATE_DIR="$(python -c "import sys; sys.path.insert(0, '.'); import config; print(config.DATASPACE)")/Registration_Evaluation"

source "${VENV_PATH}/bin/activate"

if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

echo "============================================"
echo "Registration Evaluation: Core_$(printf "%02d" $START) to Core_$(printf "%02d" $END)"
echo "Channel: ${CHANNEL}  |  MI bins: ${BINS_MI}"
echo "Script:  registration/evaluate_registration.py"
echo "Start:   $(date)"
echo "============================================"

TOTAL=$((END - START + 1))
DONE=0
FAILED=0

for i in $(seq $START $END); do

    CORE_NAME="Core_$(printf "%02d" $i)"
    echo "--------------------------------------------"
    echo "[$(date '+%H:%M:%S')] Evaluating ${CORE_NAME}..."

    python registration/evaluate_registration.py \
        --core_name "${CORE_NAME}" \
        --channel   "${CHANNEL}" \
        --bins_mi   "${BINS_MI}" \
        > "${LOG_DIR}/${CORE_NAME}.log" 2>&1

    if [ $? -eq 0 ]; then
        DONE=$((DONE + 1))
        echo "[SUCCESS] ${CORE_NAME} evaluation complete."
    else
        FAILED=$((FAILED + 1))
        echo "[ERROR]   ${CORE_NAME} failed — check ${LOG_DIR}/${CORE_NAME}.log"
        tail -n 5 "${LOG_DIR}/${CORE_NAME}.log" | sed 's/^/          /'
    fi

done

echo ""
echo "============================================"
echo "Individual evaluations done. Aggregating..."
echo "============================================"

# ─────────────────────────────────────────────────────────────────────────────
# Aggregate all per-core metrics_summary.csv files into one master table
# ─────────────────────────────────────────────────────────────────────────────
python - <<'PYEOF'
import os, sys, glob, pandas as pd

sys.path.insert(0, '.')
import config

base = os.path.join(config.DATASPACE, "Registration_Evaluation")
pattern = os.path.join(base, "Core_*", "metrics_summary.csv")
files = sorted(glob.glob(pattern))

if not files:
    print("[WARNING] No metrics_summary.csv files found — skipping aggregation.")
    sys.exit(0)

dfs = []
for f in files:
    core = os.path.basename(os.path.dirname(f))
    df = pd.read_csv(f)
    df.insert(0, "Core", core)
    dfs.append(df)

master = pd.concat(dfs, ignore_index=True)
out_path = os.path.join(base, "ALL_CORES_metrics_summary.csv")
master.to_csv(out_path, index=False)
print(f"Aggregate summary → {out_path}  ({len(dfs)} cores)")

# Also aggregate the delta files
pattern_d = os.path.join(base, "Core_*", "metrics_delta.csv")
files_d   = sorted(glob.glob(pattern_d))
if files_d:
    dfs_d = []
    for f in files_d:
        core = os.path.basename(os.path.dirname(f))
        df   = pd.read_csv(f)
        df.insert(0, "Core", core)
        dfs_d.append(df)
    master_d = pd.concat(dfs_d, ignore_index=True)
    out_d = os.path.join(base, "ALL_CORES_metrics_delta.csv")
    master_d.to_csv(out_d, index=False)
    print(f"Aggregate delta   → {out_d}")

    # Print a quick cross-core mean improvement table
    numeric_cols = [c for c in master_d.columns if "%" in c]
    if numeric_cols:
        print("\n=== Cross-Core Mean % Improvement ===")
        print(master_d.groupby("Metric")[numeric_cols].mean().round(2).to_string())

PYEOF

echo ""
echo "============================================"
echo "All done at $(date)"
echo "  Success:  ${DONE} / ${TOTAL}"
echo "  Failed:   ${FAILED} / ${TOTAL}"
echo "  Outputs:  ${AGGREGATE_DIR}/"
echo "============================================"