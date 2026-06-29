#!/usr/bin/env python3
"""
plot_landmark_violin_summary.py
================================
Generates violin (+ scatter/strip overlay) summary plots of per-landmark TRE
across the three registration pipelines (B-Spline, VALIS, RoMaV2), using the
*existing* CSV outputs already produced by run_all_landmark_accuracy.sh.

This script does NOT rerun any registration or accuracy computation. It only
reads the "_landmark_accuracy_detail.csv" files that already exist on disk
for each core, across the configured Core_START..Core_END range, and builds:

  1. A combined violin+strip plot: x = landmark_id (structure), grouped/hued
     by pipeline, y = TRE (um). Shows the full distribution per structure AND
     per-pipeline comparison -- exactly the "how do other structures perform"
     view your reviewer asked for.
  2. An overall (all-structures-pooled) violin+strip plot comparing the three
     pipelines, for a one-glance global comparison.

Usage
-----
    python plot_landmark_violin_summary.py \
        --start 1 --end 30 \
        --dataspace /path/to/dataspace \
        --out-dir /path/to/output_dir

If --dataspace is omitted, the script will try to import config.DATASPACE
from PROJECT_ROOT (same convention as run_all_landmark_accuracy.sh).

Requirements: pandas, matplotlib, seaborn
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


PIPELINE_DEFS = [
    # (label, relative_csv_template)
    ("B-Spline", "Filter_AKAZE_TissueMask_BSpline/{core}/annotation_verification_bspline/{core}_landmark_accuracy_detail.csv"),
    ("VALIS",    "VALIS_Baseline_Eval/{core}/{core}/annotation_verification_valis/{core}_VALIS_landmark_accuracy_detail.csv"),
    ("RoMaV2",   "Filter_AKAZE_RoMaV2_Linear_Warp_map/{core}/annotation_verification_Romav2/{core}_landmark_accuracy_detail.csv"),
]


def resolve_dataspace(explicit, project_root):
    if explicit:
        return explicit
    sys.path.insert(0, project_root)
    try:
        import config  # noqa
        return config.DATASPACE
    except Exception as e:
        print(f"[ERROR] Could not resolve DATASPACE automatically: {e}")
        print("        Pass --dataspace explicitly instead.")
        sys.exit(1)


def load_all(dataspace, start, end):
    """Load and concatenate all detail CSVs across cores and pipelines."""
    frames = []
    missing = []
    for i in range(start, end + 1):
        core = f"Core_{i:02d}"
        for label, template in PIPELINE_DEFS:
            path = os.path.join(dataspace, template.format(core=core))
            if not os.path.exists(path):
                missing.append((core, label, path))
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["pipeline"] = label
            df["core"] = core
            frames.append(df)

    if missing:
        print(f"[INFO] {len(missing)} expected CSV(s) not found (skipped):")
        for core, label, path in missing[:20]:
            print(f"        - {core} / {label}: {path}")
        if len(missing) > 20:
            print(f"        ... and {len(missing) - 20} more")

    if not frames:
        print("[ERROR] No CSV data found at all. Check --dataspace / --start / --end.")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def make_plots(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    pipeline_order = [p[0] for p in PIPELINE_DEFS]
    # Fresh, airy palette: teal / coral / lavender
    palette = {"B-Spline": "#2EC4B6", "VALIS": "#FF6B6B", "RoMaV2": "#A06CD5"}

    # Sort structures by overall median TRE (ascending) for readability
    order = (
        df.groupby("landmark_id")["TRE_um"]
        .median()
        .sort_values()
        .index.tolist()
    )

    # ---- 1. Per-structure violin + strip, grouped by pipeline ----
    # Horizontal orientation: structures stack on the y-axis, which scales
    # far better than cramming many x-axis labels into a fixed width.
    n_structs = len(order)
    fig_h = max(6, 0.55 * n_structs)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    sns.violinplot(
        data=df, y="landmark_id", x="TRE_um", hue="pipeline",
        order=order, hue_order=pipeline_order, palette=palette,
        ax=ax, cut=0, inner=None, linewidth=0.8, alpha=0.55,
        dodge=True, orient="h",
    )
    sns.stripplot(
        data=df, y="landmark_id", x="TRE_um", hue="pipeline",
        order=order, hue_order=pipeline_order, palette=palette,
        ax=ax, dodge=True, size=2.5, alpha=0.6, jitter=0.2,
        linewidth=0, legend=False, orient="h",
    )

    ax.set_ylabel("Landmark / structure")
    ax.set_xlabel(r"TRE ($\mu$m)")
    ax.set_title("Per-structure TRE distribution by pipeline")

    # Deduplicate legend (violin + strip both add entries)
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), title="Pipeline", loc="upper right")

    fig.tight_layout()
    per_struct_path = os.path.join(out_dir, "tre_per_structure_violin_scatter.png")
    fig.savefig(per_struct_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved: {per_struct_path}")

    # ---- 2. Overall (pooled) violin + strip across pipelines ----
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    sns.violinplot(
        data=df, x="pipeline", y="TRE_um", order=pipeline_order,
        palette=palette, ax=ax2, cut=0, inner="quartile", linewidth=1.0,
    )
    sns.stripplot(
        data=df, x="pipeline", y="TRE_um", order=pipeline_order,
        ax=ax2, color="black", size=2.5, alpha=0.35, jitter=0.25,
    )
    ax2.set_xlabel("")
    ax2.set_ylabel(r"TRE ($\mu$m)")
    ax2.set_title("Overall TRE distribution (all structures pooled)")
    fig2.tight_layout()
    overall_path = os.path.join(out_dir, "tre_overall_violin_scatter.png")
    fig2.savefig(overall_path, dpi=300)
    plt.close(fig2)
    print(f"[OK] Saved: {overall_path}")

    # ---- 3. Quick numeric summary table (for sanity check / appendix) ----
    summary = (
        df.groupby(["landmark_id", "pipeline"])["TRE_um"]
        .agg(["count", "mean", "median", "std", "max"])
        .reset_index()
        .sort_values(["landmark_id", "pipeline"])
    )
    summary_path = os.path.join(out_dir, "tre_per_structure_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--dataspace", type=str, default=None,
                         help="Root dataspace dir (overrides config.py lookup)")
    parser.add_argument("--project-root", type=str,
                         default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                         help="Project root containing config.py (used if --dataspace omitted)")
    parser.add_argument("--out-dir", type=str, default="./landmark_violin_summary",
                         help="Directory to save plots and summary CSV")
    args = parser.parse_args()

    dataspace = resolve_dataspace(args.dataspace, args.project_root)
    print(f"[INFO] Dataspace : {dataspace}")
    print(f"[INFO] Core range: Core_{args.start:02d} -> Core_{args.end:02d}")

    df = load_all(dataspace, args.start, args.end)
    print(f"[INFO] Loaded {len(df)} landmark-pair rows across "
          f"{df['core'].nunique()} cores and {df['pipeline'].nunique()} pipelines.")

    make_plots(df, args.out_dir)


if __name__ == "__main__":
    main()