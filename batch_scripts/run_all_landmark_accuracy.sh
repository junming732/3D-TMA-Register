#!/bin/bash

# =============================================================================
# run_all_landmark_accuracy.sh
# Batch execution for landmark-based TRE computation across three pipelines:
#   - Pipeline A: B-Spline
#   - Pipeline B: VALIS
#   - Pipeline C: RoMaV2
# Includes automated LaTeX table generation parsing the output CSVs.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
START=1
END=30

VENV_PATH="/home/junming/3D-TMA-Register/venv_312"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCRIPT_BSPLINE="${PROJECT_ROOT}/registration/accuracy_landmarks_bspline.py"
SCRIPT_VALIS="${PROJECT_ROOT}/registration/valis_accuracy_landmarks.py"
SCRIPT_ROMA="${PROJECT_ROOT}/registration/accuracy_landmarks_roma.py"

ANNOTATION_DIR="${PROJECT_ROOT}/annotations"

# Logging
LOG_ROOT="${PROJECT_ROOT}/log/full_pipeline"
LOG_LANDMARK="${LOG_ROOT}/landmark"

# Global flags
PIXEL_SIZE_UM=0.4961
MCLASS="all"

# -----------------------------------------------------------------------------
# SETUP & VALIDATION
# -----------------------------------------------------------------------------
source "${VENV_PATH}/bin/activate" || { echo "[ERROR] Failed to activate venv"; exit 1; }

mkdir -p "${LOG_LANDMARK}"

# Dynamically fetch DATASPACE from config
DATASPACE="$(python -c "import sys; sys.path.insert(0,'${PROJECT_ROOT}'); import config; print(config.DATASPACE)")"
if [ -z "${DATASPACE}" ]; then
    echo "[ERROR] Could not read DATASPACE from config.py"
    exit 1
fi

TOTAL=$((END - START + 1))
DONE=0
FAIL=0
SKIP=0

declare -A CORE_STATUS

echo "============================================================"
echo "  Landmark Accuracy Evaluation Pipeline"
echo "  Cores     : Core_$(printf "%02d" $START) -> Core_$(printf "%02d" $END)"
echo "  Dataspace : ${DATASPACE}"
echo "  Start time: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
for i in $(seq $START $END); do

    CORE_NAME="Core_$(printf "%02d" $i)"
    CORE_NAME_LOWER="core_$(printf "%02d" $i)"
    IDX=$((i - START + 1))

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')]  ${CORE_NAME}  (${IDX}/${TOTAL})"
    echo "------------------------------------------------------------"

    CORE_ALL_OK=1

    # 1. Prioritize merged JSON files
    ANN_JSON="${ANNOTATION_DIR}/landmark_annotation_${CORE_NAME_LOWER}.json"
    if [ ! -f "${ANN_JSON}" ]; then
        ANN_JSON="${ANNOTATION_DIR}/landmark_annotation_${CORE_NAME}.json"
    fi

    if [ ! -f "${ANN_JSON}" ]; then
        echo "  [SKIP] No annotation file found for ${CORE_NAME}"
        SKIP=$((SKIP + 3))
        CORE_STATUS[$CORE_NAME]="MISSING_JSON"
        continue
    fi

    run_pipeline() {
        local pipe_name=$1
        local py_script=$2
        local log_file="${LOG_LANDMARK}/${CORE_NAME}_${pipe_name}.log"

        echo "  [RUN] Evaluating Pipeline: ${pipe_name} (using $(basename "${ANN_JSON}")) ..."

        python "${py_script}" \
            --core_name "${CORE_NAME}" \
            --annotation_json "${ANN_JSON}" \
            --pixel_size_um "${PIXEL_SIZE_UM}" \
            --landmark_id "${MCLASS}" \
            > "${log_file}" 2>&1

        local exit_code=$?

        if [ $exit_code -ne 0 ]; then
            FAIL=$((FAIL + 1))
            CORE_ALL_OK=0
            echo "  [FAIL] ${pipe_name} failed. Check log: ${log_file}"
        else
            DONE=$((DONE + 1))
            echo "  [OK]   ${pipe_name} complete."
        fi
    }

    # Execute all three pipelines
    run_pipeline "BSpline" "${SCRIPT_BSPLINE}"
    run_pipeline "VALIS"   "${SCRIPT_VALIS}"
    run_pipeline "RoMaV2"  "${SCRIPT_ROMA}"

    if [ $CORE_ALL_OK -eq 1 ]; then
        CORE_STATUS[$CORE_NAME]="OK"
    else
        CORE_STATUS[$CORE_NAME]="PARTIAL_OR_FAIL"
    fi

    # --- LATEX SUMMARY GENERATION ---
    echo "  [LOG] Generating LaTeX summary table..."
    
    export CORE_NAME
    export DATASPACE
    export LOG_LANDMARK
    
    python - << 'EOF'
import os
import pandas as pd
import numpy as np

core_name = os.environ.get("CORE_NAME")
dataspace = os.environ.get("DATASPACE")
log_dir   = os.environ.get("LOG_LANDMARK")
out_tex   = os.path.join(log_dir, f"{core_name}_latex_summary.tex")

# Construct CSV paths based on individual pipeline logic
path_A = f"{dataspace}/Filter_AKAZE_TissueMask_BSpline/{core_name}/annotation_verification_bspline/{core_name}_landmark_accuracy_detail.csv"
path_B = f"{dataspace}/VALIS_Baseline_Eval/{core_name}/{core_name}/annotation_verification_valis/{core_name}_VALIS_landmark_accuracy_detail.csv"
path_C = f"{dataspace}/Filter_AKAZE_RoMaV2_Linear_Warp_map/{core_name}/annotation_verification_Romav2/{core_name}_landmark_accuracy_detail.csv"

def get_df(p):
    return pd.read_csv(p) if os.path.exists(p) else None

# Detail CSV (one row per consecutive pair) — for TRE metrics
dfA, dfB, dfC = get_df(path_A), get_df(path_B), get_df(path_C)

# Summary CSV (one row per mclass) — for landmark/pair counts
path_A_sum = path_A.replace('_detail.csv', '_summary.csv')
path_B_sum = path_B.replace('_detail.csv', '_summary.csv')
path_C_sum = path_C.replace('_detail.csv', '_summary.csv')
dfA_sum, dfB_sum, dfC_sum = get_df(path_A_sum), get_df(path_B_sum), get_df(path_C_sum)

if dfA is None and dfB is None and dfC is None:
    print("        No CSV outputs found to generate LaTeX.")
    import sys; sys.exit(0)

def extract_metrics(df, df_sum):
    if df is None or df.empty:
        return {'mean': np.nan, 'median': np.nan, 'max': np.nan, 'std': np.nan,
                'n_landmarks': 0, 'n_pairs': 0, 'mclass_stats': None}
    # n_landmarks = total annotated points (sum of n_slices across mclasses)
    # n_pairs     = total consecutive pairs evaluated (sum of n_pairs across mclasses)
    n_landmarks = int(df_sum['n_slices'].sum()) if df_sum is not None and 'n_slices' in df_sum else len(df)
    n_pairs     = int(df_sum['n_pairs'].sum())  if df_sum is not None and 'n_pairs'  in df_sum else len(df)
    return {
        'mean':        df['TRE_um'].mean(),
        'median':      df['TRE_um'].median(),
        'max':         df['TRE_um'].max(),
        'std':         df['TRE_um'].std(),
        'n_landmarks': n_landmarks,
        'n_pairs':     n_pairs,
        'mclass_stats': df.groupby('landmark_id')['TRE_um'].agg(mean='mean', max='max').reset_index()
    }

metA = extract_metrics(dfA, dfA_sum)
metB = extract_metrics(dfB, dfB_sum)
metC = extract_metrics(dfC, dfC_sum)

# Bolds the lowest error value (minimum is best)
def bold_min(vals, fmt="{:.2f}"):
    valid_vals = [v for v in vals if not pd.isna(v)]
    if not valid_vals: return ["--" for _ in vals]
    min_v = min(valid_vals)
    res = []
    for v in vals:
        if pd.isna(v): res.append("--")
        elif abs(v - min_v) < 1e-6: res.append(f"\\textbf{{{fmt.format(v)}}}")
        else: res.append(fmt.format(v))
    return res

# Use the pipeline with most data as reference counts
tot_landmarks = max(metA['n_landmarks'], metB['n_landmarks'], metC['n_landmarks'])

# Determine highest error & best performing structures
all_mclasses = set()
for m in [metA, metB, metC]:
    if m['mclass_stats'] is not None:
        all_mclasses.update(m['mclass_stats']['landmark_id'].unique())

mclass_agg = []
for mc in all_mclasses:
    means = []
    for m in [metA, metB, metC]:
        if m['mclass_stats'] is not None:
            row = m['mclass_stats'][m['mclass_stats']['landmark_id'] == mc]
            if not row.empty:
                means.append(row['mean'].iloc[0])
    avg_mean = np.nanmean(means) if means else np.nan
    mclass_agg.append({'mclass': mc, 'avg_mean': avg_mean})

mclass_agg.sort(key=lambda x: x['avg_mean'], reverse=True)
worst_mc = mclass_agg[0] if mclass_agg else None
best_mcs = sorted(mclass_agg, key=lambda x: x['avg_mean'])[:2]

tex = []
tex.append(f"\\begin{{table}}[htbp]")
tex.append(f"\\centering")
tex.append(f"\\caption{{TRE Summary for {core_name.replace('_', '\\_')}}}")
tex.append(f"\\label{{tab:tre_{core_name.lower().replace('_', '')}}}")
tex.append(f"\\begin{{tabular}}{{lrrr}}")
tex.append(f"\\toprule")
tex.append(f"& \\textbf{{Pipeline A}} & \\textbf{{Pipeline B}} & \\textbf{{Pipeline C}} \\\\")
tex.append(f"Metric & (Bspline) & (VALIS) & (RoMaV2) \\\\")
tex.append(f"\\midrule")
n_mclasses = len(all_mclasses)
tex.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{Global --- {n_mclasses} structures, {tot_landmarks} annotated landmarks}}}} \\\\")
tex.append(f"\\quad Mean \\gls{{tre}} ($\\mu$m)   & {' & '.join(bold_min([metA['mean'], metB['mean'], metC['mean']]))} \\\\")
tex.append(f"\\quad Median \\gls{{tre}} ($\\mu$m) & {' & '.join(bold_min([metA['median'], metB['median'], metC['median']]))} \\\\")
tex.append(f"\\quad Max \\gls{{tre}} ($\\mu$m)    & {' & '.join(bold_min([metA['max'], metB['max'], metC['max']]))} \\\\")
tex.append(f"\\quad Std \\gls{{tre}} ($\\mu$m)    & {' & '.join(bold_min([metA['std'], metB['std'], metC['std']]))} \\\\")

if worst_mc:
    tex.append(f"\\midrule")
    w_means, w_maxes = [], []
    tex.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{Highest-error structure --- landmark~{worst_mc['mclass']}}}}} \\\\")
    for m in [metA, metB, metC]:
        if m['mclass_stats'] is not None and worst_mc['mclass'] in m['mclass_stats']['landmark_id'].values:
            row = m['mclass_stats'][m['mclass_stats']['landmark_id'] == worst_mc['mclass']]
            w_means.append(row['mean'].iloc[0])
            w_maxes.append(row['max'].iloc[0])
        else:
            w_means.append(np.nan)
            w_maxes.append(np.nan)
    tex.append(f"\\quad Mean \\gls{{tre}} ($\\mu$m)   & {' & '.join(bold_min(w_means))} \\\\")
    tex.append(f"\\quad Max \\gls{{tre}} ($\\mu$m)    & {' & '.join(bold_min(w_maxes))} \\\\")

if best_mcs:
    tex.append(f"\\midrule")
    tex.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{Best-performing structures}}}} \\\\")
    for bmc in best_mcs:
        b_means = []
        for m in [metA, metB, metC]:
            if m['mclass_stats'] is not None and bmc['mclass'] in m['mclass_stats']['landmark_id'].values:
                b_means.append(m['mclass_stats'][m['mclass_stats']['landmark_id'] == bmc['mclass']]['mean'].iloc[0])
            else:
                b_means.append(np.nan)
        tex.append(f"\\quad landmark~{bmc['mclass']} mean \\gls{{tre}} ($\\mu$m) & {' & '.join(bold_min(b_means))} \\\\")

tex.append(f"\\bottomrule")
tex.append(f"\\end{{tabular}}")
tex.append(f"\\end{{table}}")

tex_str = "\n".join(tex)
with open(out_tex, "w") as f:
    f.write(tex_str + "\n")
EOF

    echo "  [OK]   LaTeX Table generated: ${LOG_LANDMARK}/${CORE_NAME}_latex_summary.tex"
done

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
MAX_RUNS=$((TOTAL * 3))

echo ""
echo "============================================================"
echo "  Landmark Accuracy Pipeline Complete -- $(date)"
echo "------------------------------------------------------------"
echo "  Cores processed : ${TOTAL}"
printf "  Pipeline-runs   : %d OK  |  %d FAILED  |  %d SKIPPED  (of %d total)\n" \
       $DONE $FAIL $SKIP $MAX_RUNS
echo "------------------------------------------------------------"

echo "  Per-core status:"
for i in $(seq $START $END); do
    CORE_NAME="Core_$(printf "%02d" $i)"
    STATUS="${CORE_STATUS[$CORE_NAME]:-UNKNOWN}"
    printf "    %-12s  %s\n" "${CORE_NAME}" "${STATUS}"
done

echo "------------------------------------------------------------"
echo "  Logs & LaTeX  : ${LOG_LANDMARK}/"
echo "============================================================"