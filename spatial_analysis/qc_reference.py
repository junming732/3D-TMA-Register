"""
qc_reference.py
====================
Shared helper so link_3d_cells.py's 2D tile QC and render_3d_cells.py's 3D
mesh QC always show the same physical cells — both within one pipeline run,
and across different registration pipelines (e.g. Valis vs Bspline vs
Romav2, whatever --input_dir_name a given run points at).

Why not just reuse the same random seed?
-----------------------------------------
cell_id_3d numbers are assigned independently by each pipeline run's graph
linking step, so the same physical nucleus can end up with a different ID
in every run. Matching by physical centroid position instead of by ID is
what makes the comparison meaningful — PROVIDED those centroids are actually
comparable across pipelines in the first place. They are not by default:
centroid_x_um/y_um are computed in each pipeline's own registered-image
space, and different pipelines crop/scale/warp differently. The one frame
every pipeline shares is the original, pre-registration per-slice image.
See raw_space_transform.py for the coordinate conversion; this module calls
it (via qc_pipeline_kind/qc_transform_dir) to convert both the reference
registry's centroids AND each candidate pipeline's centroids into that raw
frame before ever comparing them.

If pixel_size_um/section_thickness_um/qc_pipeline_kind/qc_transform_dir are
NOT all provided, this falls back to comparing centroid_x_um/y_um directly
(pre-correction behavior) — only valid for QC within a single pipeline's own
runs, not across pipelines. A clear warning is logged either way.

Behavior
--------
The first run for a given core to reach this function creates a reference
registry file — a handful of cells, their physical centroids (raw-space if
correction is active, registered-space otherwise — recorded in a
coordinate_space column so the two are never silently mixed), and which
pipeline picked them — under
    DATASPACE/QC_reference_cells/<core>_qc_reference_cells.csv
Every later run (same or different pipeline) matches against that registry
via nearest-centroid instead of drawing a fresh random sample.

Pass set_reference=True to force this run to (re)write the registry from
its own picks regardless of whether one already exists. In this pipeline,
the Snakefile passes --set_qc_reference automatically whenever
registration_variant == "valis", so Valis is always the base pipeline —
nobody has to remember to flag it by hand.

If a registered reference cell has no sufficiently close match in a given
pipeline's output (e.g. that region wasn't segmented/linked at all under
that registration), it is silently skipped for that run and logged.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REQUIRED_COLS = ['cell_id_3d', 'z_span_slices',
                  'centroid_x_um', 'centroid_y_um', 'centroid_z_um']


def _find_transform_npz(pipeline_kind, transform_dir, core_name, slice_idx):
    """Path to a given z-slice's per-pipeline transform .npz, or None if missing."""
    if pipeline_kind == 'valis':
        path = os.path.join(transform_dir, f'{core_name}_Z{slice_idx:03d}_valis_transform.npz')
        return path if os.path.isfile(path) else None
    elif pipeline_kind == 'romav2':
        for pat in (f'{core_name}_Z{slice_idx:03d}_*_deformation.npz',
                    f'{core_name}_Z{slice_idx}_*_deformation.npz'):
            matches = glob.glob(os.path.join(transform_dir, pat))
            if matches:
                return matches[0]
        return None
    else:
        raise ValueError(f"Unknown qc_pipeline_kind: {pipeline_kind!r} (expected 'valis' or 'romav2')")


def _convert_to_raw_space(df, pixel_size_um, section_thickness_um,
                           pipeline_kind, transform_dir, core_name, log):
    """
    Returns a copy of df with raw_x_um/raw_y_um columns added (converted via
    that z-slice's own transform .npz), dropping any cell whose slice has no
    transform file available (logged, not silent).
    """
    from raw_space_transform import registered_to_raw_xy

    df = df.copy().reset_index(drop=True)
    df['_slice_idx'] = (df['centroid_z_um'] / section_thickness_um).round().astype(int)

    raw_x = np.full(len(df), np.nan)
    raw_y = np.full(len(df), np.nan)
    missing_slices = []

    for slice_idx, group in df.groupby('_slice_idx'):
        npz_path = _find_transform_npz(pipeline_kind, transform_dir, core_name, int(slice_idx))
        if npz_path is None:
            missing_slices.append(int(slice_idx))
            continue
        xy_registered_px = group[['centroid_x_um', 'centroid_y_um']].values / pixel_size_um
        try:
            xy_raw_px = registered_to_raw_xy(pipeline_kind, xy_registered_px, npz_path)
        except Exception as e:
            log(f'  WARNING: raw-space conversion failed for slice_idx={slice_idx} '
                f'({type(e).__name__}: {e}) — {len(group)} cell(s) on this slice excluded.')
            continue
        xy_raw_um = xy_raw_px * pixel_size_um
        raw_x[group.index] = xy_raw_um[:, 0]
        raw_y[group.index] = xy_raw_um[:, 1]

    df['raw_x_um'] = raw_x
    df['raw_y_um'] = raw_y
    n_before = len(df)
    df = df.dropna(subset=['raw_x_um', 'raw_y_um']).reset_index(drop=True)
    n_after = len(df)

    if missing_slices:
        log(f'  Raw-space conversion: no transform .npz for z-slice(s) {sorted(set(missing_slices))} '
            f'— {n_before - n_after} cell(s) on those slices excluded from this run.')
    log(f'  Raw-space conversion: {n_after}/{n_before} cell(s) converted successfully '
        f'(pipeline_kind="{pipeline_kind}", transform_dir={transform_dir}).')
    return df


def _safe_tag(pipeline_tag: str) -> str:
    """Filesystem-safe version of a pipeline tag, for use in a filename."""
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in pipeline_tag)


def _write_match_log(registry_dir, core_name, pipeline_tag, ref_index, ref_cell_id_3d,
                      matched_cell_id_3d, dist_um, is_matched, log):
    """
    Writes <registry_dir>/<core>_qc_match_<pipeline_tag>.csv — the actual
    ref_index -> this-pipeline's-cell_id_3d lookup table. This is what you
    need to find, e.g., which RomaV2 cell_XXXXXX file corresponds to which
    Valis cell_XXXXXX file for the same physical location: match on
    ref_index (or ref_cell_id_3d, the base/Valis ID) between the two
    pipelines' correspondence files.
    """
    out = pd.DataFrame({
        'ref_index': list(ref_index),
        'ref_cell_id_3d': list(ref_cell_id_3d),
        'this_pipeline_cell_id_3d': list(matched_cell_id_3d),
        'distance_um': [round(float(d), 2) for d in dist_um],
        'matched': list(is_matched),
    })
    path = os.path.join(registry_dir, f'{core_name}_qc_match_{_safe_tag(pipeline_tag)}.csv')
    out.to_csv(path, index=False)
    return path


def get_or_match_qc_cells(core_name, df_cells, min_confirmed, n_samples,
                           pipeline_tag, dataspace, seed=0,
                           max_match_dist_um=30.0, registry_dir=None,
                           set_reference=False, logger=None,
                           pixel_size_um=None, section_thickness_um=None,
                           qc_pipeline_kind=None, qc_transform_dir=None):
    """
    Returns a list of cell_id_3d (python ints) to use for QC in THIS run,
    either by (re)writing the shared reference registry or by matching
    against it.

    df_cells must contain the columns in REQUIRED_COLS (both
    link_3d_cells.py's df_cells and render_3d_cells.py's df_stats already
    do — they're the same CSV).

    pixel_size_um, section_thickness_um, qc_pipeline_kind ('valis' or
    'romav2'), qc_transform_dir: if ALL given, cell centroids are converted
    to raw (pre-registration) image space before comparison — the only
    coordinate frame genuinely shared across different registration
    pipelines. If any is missing, falls back to comparing centroid_x_um/y_um
    directly (only valid within one pipeline's own runs).
    """
    def log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    missing = [c for c in REQUIRED_COLS if c not in df_cells.columns]
    if missing:
        raise ValueError(f'df_cells is missing required column(s): {missing}')

    use_raw_space = all(v is not None for v in
                         (pixel_size_um, section_thickness_um, qc_pipeline_kind, qc_transform_dir))
    if not use_raw_space:
        log('QC reference: raw-space coordinate correction NOT active this run '
            '(pixel_size_um / section_thickness_um / qc_pipeline_kind / qc_transform_dir '
            'not all provided) — comparing centroid_x_um/y_um directly. This is only valid '
            'for QC within a single pipeline\'s own runs; matching against a DIFFERENT '
            'pipeline\'s registry this way silently compares unrelated coordinate spaces. '
            'See raw_space_transform.py.')

    registry_dir = registry_dir or os.path.join(dataspace, 'QC_reference_cells')
    os.makedirs(registry_dir, exist_ok=True)
    registry_path = os.path.join(registry_dir, f'{core_name}_qc_reference_cells.csv')

    eligible = df_cells[df_cells['z_span_slices'] >= min_confirmed].reset_index(drop=True)
    if len(eligible) == 0:
        log('QC reference: no cells eligible (none meet min_confirmed) — skipping.')
        return []

    write_new = set_reference or not os.path.exists(registry_path)
    if write_new:
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(len(eligible), size=min(n_samples, len(eligible)), replace=False)
        picked = eligible.iloc[idx].copy()

        if use_raw_space:
            picked = _convert_to_raw_space(picked, pixel_size_um, section_thickness_um,
                                            qc_pipeline_kind, qc_transform_dir, core_name, log)
            x_col, y_col, coordinate_space = 'raw_x_um', 'raw_y_um', 'raw'
        else:
            x_col, y_col, coordinate_space = 'centroid_x_um', 'centroid_y_um', 'registered'

        ref = picked[['cell_id_3d', x_col, y_col, 'centroid_z_um', 'z_span_slices']].copy()
        ref.columns = ['cell_id_3d', 'centroid_x_um', 'centroid_y_um', 'centroid_z_um', 'z_span_slices']
        ref = ref.reset_index(drop=True)
        ref.insert(0, 'ref_index', range(len(ref)))
        ref['source_pipeline'] = pipeline_tag
        ref['coordinate_space'] = coordinate_space
        ref.to_csv(registry_path, index=False)
        reason = 'base pipeline (--set_qc_reference)' if set_reference else 'first run for this core'
        log(f'QC reference registry ({reason}): wrote {len(ref)} cells from pipeline '
            f'"{pipeline_tag}" [{coordinate_space}-space] -> {registry_path}')
        for _, row in ref.iterrows():
            log(f'    ref_index={int(row.ref_index):>3}  base cell_id_3d={int(row.cell_id_3d)}')

        # Trivial self-correspondence, written for the same reason as the
        # match log below: so every pipeline (including the base one) has a
        # ref_index -> cell_id_3d lookup file in the same place/format.
        _write_match_log(registry_dir, core_name, pipeline_tag, ref['ref_index'],
                          ref['cell_id_3d'], ref['cell_id_3d'],
                          np.zeros(len(ref)), np.ones(len(ref), dtype=bool), log)
        return ref['cell_id_3d'].astype(int).tolist()

    ref = pd.read_csv(registry_path)
    ref_space = ref['coordinate_space'].iloc[0] if 'coordinate_space' in ref.columns else 'registered'

    # Never silently compare incompatible coordinate spaces.
    if ref_space == 'raw' and not use_raw_space:
        raise RuntimeError(
            f'QC reference registry at {registry_path} was written in RAW coordinate space, '
            f'but this run did not enable raw-space correction (missing pixel_size_um / '
            f'section_thickness_um / qc_pipeline_kind / qc_transform_dir). Comparing directly '
            f'would silently match against the wrong coordinates. Provide those arguments.'
        )
    if ref_space == 'registered' and use_raw_space:
        raise RuntimeError(
            f'QC reference registry at {registry_path} is in the OLD registered-space format '
            f'(written before raw-space correction existed). Matching this run\'s raw-space '
            f'centroids against it would silently be wrong. Re-run the base pipeline with '
            f'--set_qc_reference to regenerate it in raw-space.'
        )

    if use_raw_space:
        eligible = _convert_to_raw_space(eligible, pixel_size_um, section_thickness_um,
                                          qc_pipeline_kind, qc_transform_dir, core_name, log)
        x_col, y_col = 'raw_x_um', 'raw_y_um'
    else:
        x_col, y_col = 'centroid_x_um', 'centroid_y_um'

    coords_ref = ref[['centroid_x_um', 'centroid_y_um', 'centroid_z_um']].values
    coords_now = eligible[[x_col, y_col, 'centroid_z_um']].values
    tree = cKDTree(coords_now)
    dist, nn_idx = tree.query(coords_ref, k=1)

    matched_ids, matched_cell_ids_full, is_matched = [], [], []
    n_missed = 0
    for d, i in zip(dist, nn_idx):
        if d <= max_match_dist_um:
            this_id = int(eligible.iloc[int(i)]['cell_id_3d'])
            matched_ids.append(this_id)
            matched_cell_ids_full.append(this_id)
            is_matched.append(True)
        else:
            matched_cell_ids_full.append(None)
            is_matched.append(False)
            n_missed += 1

    n_dupe = len(matched_ids) - len(set(matched_ids))
    dupe_note = f', {n_dupe} duplicate nearest-matches' if n_dupe else ''
    log(f'QC reference registry ({registry_path}): matched {len(matched_ids)}/{len(ref)} '
        f'reference cells within {max_match_dist_um} um (this pipeline="{pipeline_tag}"); '
        f'{n_missed} had no close match here{dupe_note}.')

    for idx in range(len(ref)):
        ref_idx = int(ref['ref_index'].iloc[idx])
        ref_cid = int(ref['cell_id_3d'].iloc[idx])
        if is_matched[idx]:
            log(f'    ref_index={ref_idx:>3}  base cell_id_3d={ref_cid:<8} -> '
                f'this pipeline cell_id_3d={matched_cell_ids_full[idx]:<8} '
                f'(dist={dist[idx]:.1f} um)')
        else:
            log(f'    ref_index={ref_idx:>3}  base cell_id_3d={ref_cid:<8} -> NO MATCH '
                f'(nearest was {dist[idx]:.1f} um, over the {max_match_dist_um} um cutoff)')

    match_log_path = _write_match_log(
        registry_dir, core_name, pipeline_tag, ref['ref_index'], ref['cell_id_3d'],
        matched_cell_ids_full, dist, is_matched, log,
    )
    log(f'  Per-cell correspondence also saved to -> {match_log_path}')
    return matched_ids