"""
analyze_link_metrics.py
====================
Aggregates a SINGLE registration pipeline's QC outputs from link_3d_cells.py
across all of its cores:
  - <core>/qc/all_link_metrics.csv           (per-link:  overlap_fraction,
                                               centroid_drift_px, area_ratio,
                                               out_degree_a, in_degree_b,
                                               ambiguous_link)
  - <core>/<core>_<channel>_core_summary_metrics.csv (per-core: singleton_rate,
                                               median_volume_um3, frac_volume_
                                               implausible, median_aspect_ratio,
                                               frac_multimodal_area_profile,
                                               frac_cells_with_z_gaps,
                                               frac_ambiguous_links, ...)

and reports how each metric varies core-to-core WITHIN that one pipeline:
per-core summaries, descriptive stats (median/mean/std/IQR/min/max across
cores), and plots.

Why single-pipeline, no significance testing?
-----------------------------------------------
This intentionally drops the earlier N-way (Kruskal-Wallis + Holm-corrected
Mann-Whitney) cross-pipeline comparison. Run this script once per pipeline —
outputs land inside that pipeline's own folder
(`<pipeline_dir>/Link_Metrics_Analysis/`), and every filename is still
suffixed with the pipeline name so results from separate runs never collide
and can be lined up / diffed afterwards, whether that's manually, in a
notebook, or with a separate comparison step.

Why per-core summaries, not pooled individual links?
-------------------------------------------------------
Links within one core aren't independent samples — they came from the same
tissue, the same registration run. Pooling every link across cores as if
independent lets one core with unusually many linked cells dominate the
result. This script summarizes each core down to ONE value per metric
(median by default) and treats CORES as the unit of comparison. Pooled,
link-level distributions are still plotted for exploratory context.

Usage
-----
    python analyze_link_metrics.py --pipeline CellPose_DAPI_3D_affine
    python analyze_link_metrics.py --pipeline /path/to/pipeline_dir
    python analyze_link_metrics.py --pipeline variant_1 --cores Core_09 Core_16
"""

import os
import sys
import argparse
import logging
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Per-link metrics from all_link_metrics.csv. Continuous ones are summarized
# per core with --agg (median/mean); rate ones (booleans) are always
# summarized per core as a mean, i.e. a fraction.
LINK_METRICS_CONTINUOUS = ['overlap_fraction', 'centroid_drift_px', 'area_ratio']
LINK_METRICS_RATE       = ['ambiguous_link']

LINK_METRIC_LABELS = {
    'overlap_fraction':  'Overlap fraction (higher = better)',
    'centroid_drift_px': 'Centroid drift (px, lower = better)',
    'area_ratio':        'Area ratio (closer to 1 = better)',
    'ambiguous_link':    'Fraction of links with ambiguous topology (lower = better)',
}

# Per-core metrics already aggregated inside core_summary_metrics.csv by
# link_3d_cells.py — one value per core, nothing further to aggregate here.
CORE_SUMMARY_METRICS = [
    'singleton_rate',
    'median_volume_um3',
    'frac_volume_implausible',
    'median_aspect_ratio',
    'frac_multimodal_area_profile',
    'frac_cells_with_z_gaps',
    'frac_ambiguous_links',
]

CORE_SUMMARY_LABELS = {
    'singleton_rate':               'Singleton rate (lower = better linking)',
    'median_volume_um3':            'Median cell volume (um3)',
    'frac_volume_implausible':      'Fraction volume outside plausible range',
    'median_aspect_ratio':          'Median aspect ratio (closer to 1 = better)',
    'frac_multimodal_area_profile': 'Fraction with multimodal area profile (lower = better)',
    'frac_cells_with_z_gaps':       'Fraction with Z-gaps (lower = better)',
    'frac_ambiguous_links':         'Fraction of ALL candidate links that were ambiguous',
}


# -----------------------------------------------------------------------------
# DISCOVERY + LOADING
# -----------------------------------------------------------------------------
def resolve_pipeline_dir(dataspace, pipeline):
    """Resolves a single pipeline argument (folder name under dataspace, or
    an absolute/relative path) to (pipeline_name, pipeline_dir)."""
    candidate = pipeline if os.path.isdir(pipeline) else os.path.join(dataspace, pipeline)
    if not os.path.isdir(candidate):
        raise RuntimeError(f"Pipeline directory not found: '{pipeline}' (checked '{candidate}')")
    name = os.path.basename(os.path.abspath(candidate))
    return name, os.path.abspath(candidate)


def load_pipeline_metrics(pipeline_dir, cores=None):
    """Loads and concatenates, across every core folder found under
    pipeline_dir:
      - all_link_metrics.csv       -> combined_link_df (one row per link)
      - *_core_summary_metrics.csv -> combined_core_df (one row per core)
    """
    link_frames = []
    core_frames = []
    coverage = []  # (core, n_links, n_3d_cells_total) for a coverage report

    link_csv_paths = sorted(glob.glob(os.path.join(pipeline_dir, '*', 'qc', 'all_link_metrics.csv')))
    core_csv_paths = sorted(glob.glob(os.path.join(pipeline_dir, '*', '*_core_summary_metrics.csv')))

    core_summary_by_core = {}
    for csv_path in core_csv_paths:
        core = os.path.basename(os.path.dirname(csv_path))
        if cores is not None and core not in cores:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f'  Failed to read {csv_path}: {e}')
            continue
        if df.empty:
            continue
        df['core'] = core
        core_frames.append(df)
        core_summary_by_core[core] = int(df['n_3d_cells_total'].iloc[0])

    for csv_path in link_csv_paths:
        core = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        if cores is not None and core not in cores:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f'  Failed to read {csv_path}: {e}')
            continue
        n_links = len(df)
        coverage.append((core, n_links, core_summary_by_core.get(core, None)))
        if df.empty:
            continue
        df['core'] = core
        link_frames.append(df)

    if not link_frames and not core_frames:
        raise RuntimeError(
            f"No all_link_metrics.csv or *_core_summary_metrics.csv files found under "
            f"'{pipeline_dir}'. Check --dataspace/--pipeline and that link_3d_cells.py has "
            f"been run (with --plot_qc, for the per-link file) for this pipeline/cores."
        )

    combined_link = pd.concat(link_frames, ignore_index=True) if link_frames else pd.DataFrame()
    combined_core = pd.concat(core_frames, ignore_index=True) if core_frames else pd.DataFrame()
    coverage_df = pd.DataFrame(coverage, columns=['core', 'n_links', 'n_3d_cells_total'])
    return combined_link, combined_core, coverage_df


# -----------------------------------------------------------------------------
# PER-CORE SUMMARIES + DESCRIPTIVE STATS (no cross-pipeline testing)
# -----------------------------------------------------------------------------
def summarize_link_metric_per_core(df, metric, agg):
    """One row per core: the aggregate value of `metric` for that core's
    links. `agg` is ignored for rate metrics (LINK_METRICS_RATE), which are
    always meaned (True/False -> fraction)."""
    real_agg = 'mean' if metric in LINK_METRICS_RATE else agg
    return df.groupby('core')[metric].agg(real_agg).reset_index(name=metric)


def descriptive_stats(values, metric_name):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {'metric': metric_name, 'n_cores': 0, 'median': np.nan, 'mean': np.nan,
                'std': np.nan, 'min': np.nan, 'max': np.nan, 'iqr': np.nan}
    return {
        'metric':  metric_name,
        'n_cores': len(values),
        'median':  float(np.median(values)),
        'mean':    float(np.mean(values)),
        'std':     float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'min':     float(np.min(values)),
        'max':     float(np.max(values)),
        'iqr':     float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
def plot_link_metric(df, per_core_summary, metric, pipeline_name, out_dir):
    """Left: pooled link-level distribution across all cores (exploratory).
    Right: per-core aggregate value, one bar per core (the actual unit used
    for the descriptive stats)."""
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.0 * len(per_core_summary) + 6), 5))

    pooled_vals = df[metric].dropna().values.astype(float)
    axes[0].boxplot(pooled_vals, tick_labels=[pipeline_name], showfliers=False)
    axes[0].set_title(f'{LINK_METRIC_LABELS.get(metric, metric)}\n(pooled links, exploratory only)')
    axes[0].set_ylabel(metric)

    cores_sorted = per_core_summary.sort_values('core').copy()
    cores_sorted[metric] = cores_sorted[metric].astype(float)
    axes[1].bar(cores_sorted['core'], cores_sorted[metric], color='steelblue')
    axes[1].axhline(cores_sorted[metric].median(), color='black', linestyle='--', linewidth=1,
                     label='median across cores')
    axes[1].set_title(f'{LINK_METRIC_LABELS.get(metric, metric)}\n(per-core {("mean/rate" if metric in LINK_METRICS_RATE else "aggregate")})')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel(metric)
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{metric}_{pipeline_name}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_core_summary_metric(core_df, metric, pipeline_name, out_dir):
    """One bar per core for a metric that's already per-core (from
    core_summary_metrics.csv)."""
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(core_df) + 3), 5))
    cores_sorted = core_df.sort_values('core')
    ax.bar(cores_sorted['core'], cores_sorted[metric], color='indianred')
    ax.axhline(cores_sorted[metric].median(), color='black', linestyle='--', linewidth=1,
               label='median across cores')
    ax.set_title(f'{CORE_SUMMARY_LABELS.get(metric, metric)}\n{pipeline_name}')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel(metric)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{metric}_{pipeline_name}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataspace', type=str, default=config.DATASPACE)
    parser.add_argument('--pipeline', type=str, required=True,
                        help='A single pipeline folder name (under --dataspace) or absolute path.')
    parser.add_argument('--cores', nargs='*', default=None,
                        help='Restrict to these core names. Default: use every core found.')
    parser.add_argument('--agg', type=str, default='median', choices=['median', 'mean'],
                        help='How to summarize each core down to one value for the continuous '
                             'per-link metrics before reporting descriptive stats (default: median).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save aggregated CSVs and plots '
                             '(default: <pipeline_dir>/Link_Metrics_Analysis, i.e. inside the '
                             'pipeline folder itself).')
    args = parser.parse_args()

    pipeline_name, pipeline_dir = resolve_pipeline_dir(args.dataspace, args.pipeline)
    output_dir = args.output_dir or os.path.join(pipeline_dir, 'Link_Metrics_Analysis')
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f'Dataspace     : {args.dataspace}')
    logger.info(f'Pipeline      : {pipeline_name}  ({pipeline_dir})')
    logger.info(f'Cores         : {"ALL" if args.cores is None else args.cores}')
    logger.info(f'Per-core agg  : {args.agg}')
    logger.info(f'Output        : {output_dir}')
    logger.info('=' * 70)

    combined_link, combined_core, coverage_df = load_pipeline_metrics(pipeline_dir, cores=args.cores)

    coverage_path = os.path.join(output_dir, f'coverage_{pipeline_name}.csv')
    coverage_df.to_csv(coverage_path, index=False)
    logger.info(f'Coverage (core x n_links x n_3d_cells) -> {coverage_path}')
    logger.info('\n' + coverage_df.to_string(index=False))

    if not combined_link.empty:
        link_path = os.path.join(output_dir, f'combined_link_metrics_{pipeline_name}.csv')
        combined_link.to_csv(link_path, index=False)
        logger.info(f'Combined raw link metrics ({len(combined_link)} rows) -> {link_path}')

    if not combined_core.empty:
        core_path = os.path.join(output_dir, f'core_summary_metrics_{pipeline_name}.csv')
        combined_core.to_csv(core_path, index=False)
        logger.info(f'Combined per-core summary metrics ({len(combined_core)} rows) -> {core_path}')

    logger.info('=' * 70)
    all_stats = []

    # --- Per-link metrics (continuous + rate), aggregated to one value/core ---
    if not combined_link.empty:
        for metric in LINK_METRICS_CONTINUOUS + LINK_METRICS_RATE:
            if metric not in combined_link.columns:
                logger.warning(f'  Column "{metric}" not found in combined link metrics — skipping.')
                continue
            per_core = summarize_link_metric_per_core(combined_link, metric, agg=args.agg)
            summary_path = os.path.join(output_dir, f'{metric}_per_core_summary_{pipeline_name}.csv')
            per_core.to_csv(summary_path, index=False)
            stats = descriptive_stats(per_core[metric].values, metric)
            all_stats.append(stats)
            logger.info(f'--- {metric} (n_cores={stats["n_cores"]}) ---')
            logger.info(f'  median={stats["median"]:.3g}  mean={stats["mean"]:.3g}  '
                        f'std={stats["std"]:.3g}  iqr={stats["iqr"]:.3g}  '
                        f'range=[{stats["min"]:.3g}, {stats["max"]:.3g}]')
            plot_path = plot_link_metric(combined_link, per_core, metric, pipeline_name, output_dir)
            logger.info(f'  Per-core summary -> {summary_path}')
            logger.info(f'  Plot -> {plot_path}')
            logger.info('-' * 70)
    else:
        logger.warning('No per-link metrics found (all_link_metrics.csv) — was link_3d_cells.py '
                        'run with --plot_qc for this pipeline? Skipping link-level metrics.')

    # --- Per-core metrics, already one value/core (from core_summary_metrics.csv) ---
    if not combined_core.empty:
        for metric in CORE_SUMMARY_METRICS:
            if metric not in combined_core.columns:
                logger.warning(f'  Column "{metric}" not found in core summary metrics — skipping.')
                continue
            stats = descriptive_stats(combined_core[metric].values, metric)
            all_stats.append(stats)
            logger.info(f'--- {metric} (n_cores={stats["n_cores"]}) ---')
            logger.info(f'  median={stats["median"]:.3g}  mean={stats["mean"]:.3g}  '
                        f'std={stats["std"]:.3g}  iqr={stats["iqr"]:.3g}  '
                        f'range=[{stats["min"]:.3g}, {stats["max"]:.3g}]')
            plot_path = plot_core_summary_metric(combined_core, metric, pipeline_name, output_dir)
            logger.info(f'  Plot -> {plot_path}')
            logger.info('-' * 70)
    else:
        logger.warning('No core-level summary metrics found (*_core_summary_metrics.csv) — '
                        'skipping core-level metrics.')

    if all_stats:
        stats_path = os.path.join(output_dir, f'descriptive_stats_{pipeline_name}.csv')
        pd.DataFrame(all_stats).to_csv(stats_path, index=False)
        logger.info(f'All descriptive stats (across cores, this pipeline) -> {stats_path}')

    logger.info('=' * 70)
    logger.info('Done. To compare against another pipeline, run this script again with '
                '--pipeline <other_pipeline> and diff/join the resulting *_<pipeline>.csv files.')


if __name__ == '__main__':
    main()