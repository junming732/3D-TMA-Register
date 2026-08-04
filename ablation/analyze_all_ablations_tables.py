"""
analyze_all_ablations_tables.py
======================================================================
Comprehensive tabular analysis of all registration ablation experiments.

Replaces graphical plots with clean, tabular output covering:
  1. Target Registration Error (TRE mean & median in pixels)
  2. Normalized Cross-Correlation (NCC warp & affine)
  3. Coverage & Confidence metrics
  4. Execution runtime and cost multipliers

Usage:
    python analyze_all_ablations_tables.py
    python analyze_all_ablations_tables.py --out_md report.md
"""

import os
import argparse
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon, friedmanchisquare
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_paired_stats(df, base_col, cand_col, lower_is_better=True):
    """
    Computes paired delta, win/loss counts, mean/median deltas, and Wilcoxon p-value.
    For NCC, lower_is_better=True (more negative = better).
    For TRE, lower_is_better=True (fewer pixels error = better).
    """
    sub = df.dropna(subset=[base_col, cand_col]).copy()
    if len(sub) == 0:
        return None

    delta = sub[cand_col] - sub[base_col]
    
    if lower_is_better:
        n_better = int((delta < -1e-6).sum())
        n_worse  = int((delta > 1e-6).sum())
    else:
        n_better = int((delta > 1e-6).sum())
        n_worse  = int((delta < -1e-6).sum())
        
    n_tied = len(sub) - n_better - n_worse

    p_val = np.nan
    if HAVE_SCIPY and len(sub) >= 5 and not np.allclose(sub[base_col], sub[cand_col]):
        try:
            _, p_val = wilcoxon(sub[base_col], sub[cand_col])
        except ValueError:
            pass

    return {
        "n_pairs": len(sub),
        "base_mean": sub[base_col].mean(),
        "cand_mean": sub[cand_col].mean(),
        "base_median": sub[base_col].median(),
        "cand_median": sub[cand_col].median(),
        "mean_delta": delta.mean(),
        "median_delta": delta.median(),
        "n_better": n_better,
        "n_worse": n_worse,
        "n_tied": n_tied,
        "p_value": p_val
    }


def format_p_value(p):
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return f"{p:.2e} (***)"
    if p < 0.01:
        return f"{p:.4f} (**)"
    if p < 0.05:
        return f"{p:.4f} (*)"
    return f"{p:.4f} (ns)"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1-3 Analysis: Solvers, Tukey, Downsampling
# ─────────────────────────────────────────────────────────────────────────────

def analyze_step123(df):
    if df is None or len(df) == 0:
        return "### Steps 1-3 (Solvers & Detection)\nNo data found.\n\n"

    out = []
    out.append("## Steps 1-3: Solver, Filter, & Detection Ablation\n")

    # 1. MAGSAC vs RANSAC
    if "RANSAC_ncc_affine" in df.columns and "USAC_MAGSAC_ncc_affine" in df.columns:
        stats_ncc = compute_paired_stats(df, "RANSAC_ncc_affine", "USAC_MAGSAC_ncc_affine", lower_is_better=True)
        
        # Check for TRE columns if present
        has_tre = "RANSAC_tre_mean_px" in df.columns and "USAC_MAGSAC_tre_mean_px" in df.columns
        stats_tre = compute_paired_stats(df, "RANSAC_tre_mean_px", "USAC_MAGSAC_tre_mean_px", lower_is_better=True) if has_tre else None

        table_data = []
        if stats_ncc:
            table_data.append({
                "Metric": "NCC Affine (more negative = better)",
                "RANSAC": f"{stats_ncc['base_mean']:.5f}",
                "MAGSAC": f"{stats_ncc['cand_mean']:.5f}",
                "Mean Delta": f"{stats_ncc['mean_delta']:.5f}",
                "Better / Worse / Tied": f"{stats_ncc['n_better']} / {stats_ncc['n_worse']} / {stats_ncc['n_tied']}",
                "p-value": format_p_value(stats_ncc['p_value'])
            })
        if stats_tre:
            table_data.append({
                "Metric": "TRE Mean (px, lower = better)",
                "RANSAC": f"{stats_tre['base_mean']:.3f} px",
                "MAGSAC": f"{stats_tre['cand_mean']:.3f} px",
                "Mean Delta": f"{stats_tre['mean_delta']:+.3f} px",
                "Better / Worse / Tied": f"{stats_tre['n_better']} / {stats_tre['n_worse']} / {stats_tre['n_tied']}",
                "p-value": format_p_value(stats_tre['p_value'])
            })

        if table_data:
            out.append("### 1. USAC_MAGSAC vs Standard RANSAC")
            out.append(pd.DataFrame(table_data).to_markdown(index=False))
            out.append("\n")

    # 2. Downsampled vs Full-Res
    if "downsampled_ncc_affine" in df.columns and "RANSAC_ncc_affine" in df.columns:
        stats_d = compute_paired_stats(df, "RANSAC_ncc_affine", "downsampled_ncc_affine", lower_is_better=True)
        has_time = "fullres_detect_time_s" in df.columns and "downsampled_detect_time_s" in df.columns
        
        table_data = []
        if stats_d:
            table_data.append({
                "Comparison": "Full-Res vs Downsampled AKAZE",
                "Full-Res NCC": f"{stats_d['base_mean']:.5f}",
                "Downsampled NCC": f"{stats_d['cand_mean']:.5f}",
                "NCC Delta": f"{stats_d['mean_delta']:.5f}",
                "p-value": format_p_value(stats_d['p_value'])
            })
        
        if has_time:
            t_sub = df.dropna(subset=["fullres_detect_time_s", "downsampled_detect_time_s"])
            speedup = t_sub["fullres_detect_time_s"].mean() / max(t_sub["downsampled_detect_time_s"].mean(), 1e-6)
            table_data[0]["Speedup"] = f"{speedup:.1f}x faster"

        if table_data:
            out.append("### 2. Downsampled Keypoint Detection Tradeoff")
            out.append(pd.DataFrame(table_data).to_markdown(index=False))
            out.append("\n")

    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 Analysis: Tiled vs Whole-Image RoMaV2
# ─────────────────────────────────────────────────────────────────────────────

def analyze_step4(df):
    if df is None or len(df) == 0:
        return "### Step 4 (Tiled RoMaV2)\nNo data found.\n\n"

    out = []
    out.append("## Step 4: Tiled vs. Whole-Image RoMaV2 Ablation\n")

    stats_ncc = compute_paired_stats(df, "whole_ncc_warp", "tiled_ncc_warp", lower_is_better=True)
    stats_tre_mean = compute_paired_stats(df, "whole_tre_mean_px", "tiled_tre_mean_px", lower_is_better=True)
    stats_tre_med  = compute_paired_stats(df, "whole_tre_median_px", "tiled_tre_median_px", lower_is_better=True)

    t_whole = df["whole_time_s"].mean() if "whole_time_s" in df.columns else np.nan
    t_tiled = df["tiled_time_s"].mean() if "tiled_time_s" in df.columns else np.nan
    cov_whole = df["whole_coverage_pct"].mean() if "whole_coverage_pct" in df.columns else np.nan
    cov_tiled = df["tiled_coverage_pct"].mean() if "tiled_coverage_pct" in df.columns else np.nan

    table_rows = []

    if stats_ncc:
        table_rows.append({
            "Metric": "NCC Warp (more negative = better)",
            "Whole-Image": f"{stats_ncc['base_mean']:.5f}",
            "Tiled": f"{stats_ncc['cand_mean']:.5f}",
            "Delta (Tiled - Whole)": f"{stats_ncc['mean_delta']:+.5f}",
            "Tiled Wins / Losses / Ties": f"{stats_ncc['n_better']} / {stats_ncc['n_worse']} / {stats_ncc['n_tied']}",
            "p-value": format_p_value(stats_ncc['p_value'])
        })

    if stats_tre_mean:
        table_rows.append({
            "Metric": "TRE Mean (px, lower = better)",
            "Whole-Image": f"{stats_tre_mean['base_mean']:.3f} px",
            "Tiled": f"{stats_tre_mean['cand_mean']:.3f} px",
            "Delta (Tiled - Whole)": f"{stats_tre_mean['mean_delta']:+.3f} px",
            "Tiled Wins / Losses / Ties": f"{stats_tre_mean['n_better']} / {stats_tre_mean['n_worse']} / {stats_tre_mean['n_tied']}",
            "p-value": format_p_value(stats_tre_mean['p_value'])
        })

    if stats_tre_med:
        table_rows.append({
            "Metric": "TRE Median (px, lower = better)",
            "Whole-Image": f"{stats_tre_med['base_median']:.3f} px",
            "Tiled": f"{stats_tre_med['cand_median']:.3f} px",
            "Delta (Tiled - Whole)": f"{stats_tre_med['median_delta']:+.3f} px",
            "Tiled Wins / Losses / Ties": f"{stats_tre_med['n_better']} / {stats_tre_med['n_worse']} / {stats_tre_med['n_tied']}",
            "p-value": format_p_value(stats_tre_med['p_value'])
        })

    if not np.isnan(cov_whole):
        table_rows.append({
            "Metric": "Coverage %",
            "Whole-Image": f"{cov_whole:.1f}%",
            "Tiled": f"{cov_tiled:.1f}%",
            "Delta (Tiled - Whole)": f"{cov_tiled - cov_whole:+.1f}%",
            "Tiled Wins / Losses / Ties": "N/A",
            "p-value": "N/A"
        })

    if not np.isnan(t_whole):
        cost_mult = t_tiled / max(t_whole, 1e-6)
        table_rows.append({
            "Metric": "Mean Runtime (s)",
            "Whole-Image": f"{t_whole:.2f} s",
            "Tiled": f"{t_tiled:.2f} s",
            "Delta (Tiled - Whole)": f"{cost_mult:.1f}x cost multiplier",
            "Tiled Wins / Losses / Ties": "N/A",
            "p-value": "N/A"
        })

    out.append(pd.DataFrame(table_rows).to_markdown(index=False))
    out.append("\n")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 Analysis: ROMA_MODE Sweep
# ─────────────────────────────────────────────────────────────────────────────

def analyze_step5(df):
    if df is None or len(df) == 0:
        return "### Step 5 (ROMA_MODE)\nNo data found.\n\n"

    out = []
    out.append("## Step 5: ROMA_MODE Channel Fusion Sweep\n")

    ncc_cols = [c for c in df.columns if c.endswith("_ncc_warp")]
    modes = [c[:-len("_ncc_warp")] for c in ncc_cols]

    if "l0_only_tre_mean_px" in df.columns:
        l0_valid = df["l0_only_tre_mean_px"].dropna()
        out.append(f"> **L0 Affine Inlier Stability Reference:** Mean TRE = {l0_valid.mean():.3f} px | Median TRE = {l0_valid.median():.3f} px\n")

    summary_rows = []
    has_baseline = "ck_only_ncc_warp" in df.columns

    for mode in modes:
        ncc_valid = df[f"{mode}_ncc_warp"].dropna()
        tre_mean_valid = df[f"{mode}_tre_mean_px"].dropna() if f"{mode}_tre_mean_px" in df.columns else pd.Series()
        tre_med_valid  = df[f"{mode}_tre_median_px"].dropna() if f"{mode}_tre_median_px" in df.columns else pd.Series()
        time_valid = df[f"{mode}_time_s"].dropna() if f"{mode}_time_s" in df.columns else pd.Series()
        cov_valid  = df[f"{mode}_coverage_pct"].dropna() if f"{mode}_coverage_pct" in df.columns else pd.Series()

        # Comparisons vs baseline (ck_only)
        ncc_vs_base = compute_paired_stats(df, "ck_only_ncc_warp", f"{mode}_ncc_warp", lower_is_better=True) if (has_baseline and mode != "ck_only") else None
        tre_vs_base = compute_paired_stats(df, "ck_only_tre_mean_px", f"{mode}_tre_mean_px", lower_is_better=True) if (has_baseline and mode != "ck_only" and "ck_only_tre_mean_px" in df.columns) else None

        # Agreement check between NCC and TRE
        agreement = "BASELINE"
        if mode != "ck_only":
            if ncc_vs_base and tre_vs_base:
                ncc_better = ncc_vs_base["n_better"] > ncc_vs_base["n_worse"]
                tre_better = tre_vs_base["n_better"] > tre_vs_base["n_worse"]
                if ncc_better == tre_better:
                    agreement = "AGREE (" + ("BETTER" if ncc_better else "WORSE") + ")"
                else:
                    agreement = "DISAGREE (NCC=" + ("BETTER" if ncc_better else "WORSE") + ", TRE=" + ("BETTER" if tre_better else "WORSE") + ")"
            elif ncc_vs_base:
                agreement = "NCC ONLY"

        summary_rows.append({
            "Mode": mode,
            "Pairs (N)": len(ncc_valid),
            "NCC Warp (Mean)": f"{ncc_valid.mean():.5f}" if len(ncc_valid) else "N/A",
            "TRE Mean (px)": f"{tre_mean_valid.mean():.3f}" if len(tre_mean_valid) else "N/A",
            "TRE Median (px)": f"{tre_med_valid.median():.3f}" if len(tre_med_valid) else "N/A",
            "TRE Delta vs Baseline": f"{tre_vs_base['mean_delta']:+.3f} px" if tre_vs_base else ("0.000 px" if mode == "ck_only" else "N/A"),
            "TRE p-value": format_p_value(tre_vs_base['p_value']) if tre_vs_base else "N/A",
            "Coverage %": f"{cov_valid.mean():.1f}%" if len(cov_valid) else "N/A",
            "Time (s)": f"{time_valid.mean():.2f}s" if len(time_valid) else "N/A",
            "Verdict vs Baseline": agreement
        })

    summary_df = pd.DataFrame(summary_rows)
    # Sort table by TRE Mean (lowest error first)
    if "TRE Mean (px)" in summary_df.columns:
        summary_df["_sort_key"] = pd.to_numeric(summary_df["TRE Mean (px)"].str.replace(" px", ""), errors="coerce")
        summary_df = summary_df.sort_values("_sort_key").drop(columns=["_sort_key"])

    out.append(summary_df.to_markdown(index=False))
    out.append("\n")

    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Main Routine
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tabular report generator for registration ablations.")
    parser.add_argument("--ransac_csv", type=str, default="ransac_vs_magsac_ablation.csv")
    parser.add_argument("--tiled_csv",  type=str, default="tre_tiled_romav2_ablation.csv")
    parser.add_argument("--mode_csv",   type=str, default="tre_roma_mode_ablation.csv")
    parser.add_argument("--out_md",     type=str, default="ablation_summary_report.md")
    args = parser.parse_args()

    df1 = pd.read_csv(args.ransac_csv) if os.path.exists(args.ransac_csv) else None
    df2 = pd.read_csv(args.tiled_csv)  if os.path.exists(args.tiled_csv)  else None
    df3 = pd.read_csv(args.mode_csv)   if os.path.exists(args.mode_csv)   else None

    report = []
    report.append("# Comprehensive Registration Pipeline Ablation Report\n")
    report.append("This document summarizes all ablation steps across NCC alignment, Target Registration Error (TRE in pixels), coverage, and processing speeds.\n")

    report.append(analyze_step123(df1))
    report.append(analyze_step4(df2))
    report.append(analyze_step5(df3))

    full_report_text = "\n".join(report)

    # Output to Console
    print("\n" + "=" * 80)
    print(full_report_text)
    print("=" * 80 + "\n")

    # Output to Markdown File
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write(full_report_text)
    print(f"Summary report written to: {os.path.abspath(args.out_md)}")


if __name__ == "__main__":
    main()