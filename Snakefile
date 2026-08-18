"""
Snakefile
=========
Orchestrates the full 3D-TMA pipeline: registration -> CellPose segmentation
-> mask warp -> denoising -> 3D linking -> phenotyping -> cell-type assignment
-> 2D/3D TME comparison -> aggregation.
"""

import os
import sys

configfile: "config.yaml"

# ── Load DATASPACE / VENV_PATH from the pipeline's own config.py ────────────
sys.path.insert(0, workflow.basedir)
import config as pcfg

DATASPACE = pcfg.DATASPACE.rstrip('/')
PYTHON    = os.path.join(pcfg.VENV_PATH, "bin", "python")
SCRIPTS   = workflow.basedir   

# Set global temp directory via environment variable (compatible with all Snakemake versions)
os.environ["TMPDIR"] = os.path.join(getattr(pcfg, "WORKSPACE", "/tmp"), "snakemake_tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# ── Core list ────────────────────────────────────────────────────────────────
_excluded = {f"Core_{x:02d}" for x in config.get("excluded_cores", [])}
CORES = [
    f"Core_{i:02d}"
    for i in range(config["core_start"], config["core_end"] + 1)
    if f"Core_{i:02d}" not in _excluded
]

# ── Registration variant ──────────────────────────────────────────────────────
REG_VARIANT = config["registration_variant"]
REG_CFG     = config["registration"][REG_VARIANT]
TAG         = REG_CFG.get("tag", REG_VARIANT)

# ── Logging Base Directory ───────────────────────────────────────────────────
LOG_BASE = f"log/{TAG}"

# ── Evaluation Configuration & Variant Mapping ────────────────────────────────
EVAL_CONFIGS = {
    "romav2": {
        "script": f"{SCRIPTS}/evaluation/accuracy_landmarks_deform.py",
        "pipeline_flag": "--pipeline roma",
        "detail_path": "{dataspace}/{work_dir}/{core}/annotation_verification_Romav2/{core}_landmark_accuracy_detail.csv",
    },
    "bspline": {
        "script": f"{SCRIPTS}/evaluation/accuracy_landmarks_deform.py",
        "pipeline_flag": "--pipeline bspline",
        "detail_path": "{dataspace}/{work_dir}/{core}/annotation_verification_bspline/{core}_landmark_accuracy_detail.csv",
    },
    "valis": {
        "script": f"{SCRIPTS}/evaluation/valis_accuracy_landmarks.py",
        "pipeline_flag": "", 
        "detail_path": "{dataspace}/{work_dir}/{core}/{core}/annotation_verification_valis/{core}_VALIS_landmark_accuracy_detail.csv",
    },
}

ACTIVE_EVAL = EVAL_CONFIGS.get(REG_VARIANT, EVAL_CONFIGS["romav2"])

# Dynamically discover which cores have annotation JSON files available
ANNOTATION_DIR = os.path.join(workflow.basedir, "annotations")
EVALUABLE_CORES = []
CORE_TO_JSON = {}

for core in CORES:
    json_lower = os.path.join(ANNOTATION_DIR, f"landmark_annotation_{core.lower()}.json")
    json_exact = os.path.join(ANNOTATION_DIR, f"landmark_annotation_{core}.json")
    
    if os.path.exists(json_lower):
        EVALUABLE_CORES.append(core)
        CORE_TO_JSON[core] = json_lower
    elif os.path.exists(json_exact):
        EVALUABLE_CORES.append(core)
        CORE_TO_JSON[core] = json_exact

def _with_tag(cfg_block):
    resolved = dict(cfg_block)
    for key, value in resolved.items():
        if key.endswith("_dir_name") and isinstance(value, str) and "{tag}" in value:
            resolved[key] = value.format(tag=TAG)
    return resolved

CELLPOSE  = _with_tag(config["cellpose"])
DENOISE   = _with_tag(config["denoise"])
LINK3D    = _with_tag(config["link_3d"])
PHENO     = _with_tag(config["phenotype"])
ASSIGN    = config["assign_phenotypes"]
TME       = _with_tag(config["tme_comparison"])
AGG       = config["aggregate"]
RENDER3D  = config["render_3d"]


# =============================================================================
# TARGET RULE
# =============================================================================
def all_targets():
    # 1. Main TME output
    targets = [f"{DATASPACE}/{TME['output_dir_name']}/Aggregate/aggregate_summary.csv"]
    
    # 2. Render 3D QC plots (always enabled now)
    targets += [
        f"{DATASPACE}/{LINK3D['output_dir_name']}/{core}/qc/render_3d_qc/.done"
        for core in CORES
    ]
        
    # 3. Target CSV paths for cores with available annotations
    targets += [
        ACTIVE_EVAL["detail_path"].format(
            dataspace=DATASPACE,
            work_dir=REG_CFG["output_dir_name"],
            core=core
        )
        for core in EVALUABLE_CORES
    ]
    return targets

rule all:
    input:
        all_targets()


# =============================================================================
# STAGE 1 — REGISTRATION
# =============================================================================
rule registration:
    output:
        vol = f"{DATASPACE}/{REG_CFG['output_dir_name']}/{{core}}/{{core}}{REG_CFG['vol_filename_suffix']}",
        deform_dir = directory(f"{DATASPACE}/{REG_CFG['output_dir_name']}/{{core}}/deformation_maps"),
    params:
        script = f"{SCRIPTS}/{REG_CFG['script']}",
        extra  = REG_CFG["extra_args"],
    log:
        f"{LOG_BASE}/registration/{{core}}.log"
    threads: 4
    shell:
        "{PYTHON} {params.script} --core_name {wildcards.core} {params.extra} > {log} 2>&1"


# =============================================================================
# STAGE 1b — LANDMARK ACCURACY EVALUATION 
# =============================================================================
rule evaluate_landmarks:
    input:
        vol = f"{DATASPACE}/{REG_CFG['output_dir_name']}/{{core}}/{{core}}{REG_CFG['vol_filename_suffix']}",
    output:
        detail_csv = ACTIVE_EVAL["detail_path"].format(
            dataspace=DATASPACE,
            work_dir=REG_CFG["output_dir_name"],
            core="{core}"
        ),
    params:
        script          = ACTIVE_EVAL["script"],
        pipe_flag       = ACTIVE_EVAL["pipeline_flag"],
        work_output_dir = REG_CFG["output_dir_name"],
        pixel_size_um   = DENOISE.get("pixel_um", 0.4961),
        annotation_json = lambda wildcards: CORE_TO_JSON[wildcards.core],
    log:
        f"{LOG_BASE}/evaluate_landmarks/{{core}}.log"
    threads: 1
    shell:
        "{PYTHON} {params.script} --core_name {wildcards.core} "
        "{params.pipe_flag} "
        "--annotation_json {params.annotation_json} "
        "--pixel_size_um {params.pixel_size_um} "
        "--work_output_dir {params.work_output_dir} "
        " > {log} 2>&1"


# =============================================================================
# STAGE 2a — CELLPOSE SEGMENTATION 
# =============================================================================
rule cellpose_segmentation:
    output:
        mask_dir = directory(f"{DATASPACE}/{CELLPOSE['mask_dir_name']}/{{core}}"),
    params:
        script = f"{SCRIPTS}/spatial_analysis/cellpose_segmentation.py",
        gpu_flag = "--use_gpu" if CELLPOSE["use_gpu"] else "",
        qc_flag  = "--plot_qc" if CELLPOSE["plot_qc"] else "",
    log:
        "log/shared_cellpose/cellpose_{core}.log"
    threads: 2
    resources:
        gpu = 1 if CELLPOSE["use_gpu"] else 0,
    shell:
        "{PYTHON} {params.script} --core_name {wildcards.core} "
        "{params.gpu_flag} {params.qc_flag} > {log} 2>&1"


# =============================================================================
# STAGE 2b — WARP CELLPOSE MASKS
# =============================================================================
def get_warp_deform_dir(wildcards):
    if REG_VARIANT == "valis":
        return f"{DATASPACE}/{REG_CFG['output_dir_name']}/{wildcards.core}/{wildcards.core}/data"
    else:
        return f"{DATASPACE}/{REG_CFG['output_dir_name']}/{wildcards.core}/deformation_maps"

def get_warp_script(wildcards):
    if REG_VARIANT == "valis":
        return f"{SCRIPTS}/spatial_analysis/warp_cellpose_masks_valis.py"
    else:
        return f"{SCRIPTS}/spatial_analysis/warp_cellpose_masks.py"

rule warp_cellpose_masks:
    input:
        mask_dir   = f"{DATASPACE}/{CELLPOSE['mask_dir_name']}/{{core}}",
        deform_dir = get_warp_deform_dir,
    output:
        warped_dir = directory(f"{DATASPACE}/{CELLPOSE['warped_dir_name']}/{{core}}"),
    params:
        script  = get_warp_script,
        qc_flag = "--plot_qc" if CELLPOSE["plot_qc"] else "",
    log:
        f"{LOG_BASE}/warp_cellpose_masks/{{core}}.log"
    threads: 2
    shell:
        "{PYTHON} {params.script} --core_name {wildcards.core} "
        "--mask_dir {input.mask_dir} --deform_dir {input.deform_dir} "
        "--out_dir {output.warped_dir} {params.qc_flag} > {log} 2>&1"


# =============================================================================
# STAGE 3 — DENOISING
# =============================================================================
rule denoise_volume:
    input:
        vol = f"{DATASPACE}/{REG_CFG['output_dir_name']}/{{core}}/{{core}}{REG_CFG['vol_filename_suffix']}",
    output:
        denoised = f"{DATASPACE}/{DENOISE['output_dir_name']}/{{core}}/{{core}}_denoised.ome.tif",
    params:
        script            = f"{SCRIPTS}/spatial_analysis/denoise_volume.py",
        pixel_um          = DENOISE["pixel_um"],
        dust_pct          = DENOISE["dust_pct"],
        workers           = DENOISE["workers"],
        input_dir_name    = REG_CFG["output_dir_name"],
        input_file_suffix = REG_CFG["vol_filename_suffix"],
        output_dir_name   = DENOISE["output_dir_name"],
    log:
        f"{LOG_BASE}/denoise_volume/{{core}}.log"
    threads: lambda wc: DENOISE["workers"]
    shell:
        "{PYTHON} {params.script} --core_name {wildcards.core} "
        "--pixel_um {params.pixel_um} --dust_pct {params.dust_pct} "
        "--workers {params.workers} --overwrite --plot_qc "
        "--input_dir_name {params.input_dir_name} "
        "--input_file_suffix {params.input_file_suffix} "
        "--output_dir_name {params.output_dir_name} > {log} 2>&1"


# =============================================================================
# STAGE 4 — 3D LINKING
# =============================================================================
rule link_3d_cells:
    input:
        warped_dir = f"{DATASPACE}/{CELLPOSE['warped_dir_name']}/{{core}}",
        denoised   = f"{DATASPACE}/{DENOISE['output_dir_name']}/{{core}}/{{core}}_denoised.ome.tif",
    output:
        stats  = f"{DATASPACE}/{LINK3D['output_dir_name']}/{{core}}/{{core}}_DAPI_3d_stats.csv",
        map_   = f"{DATASPACE}/{LINK3D['output_dir_name']}/{{core}}/{{core}}_DAPI_2d_to_3d_map.csv",
        labels = f"{DATASPACE}/{LINK3D['output_dir_name']}/{{core}}/{{core}}_DAPI_3d_labels.tif",
    params:
        script  = f"{SCRIPTS}/spatial_analysis/link_3d_cells.py",
        L       = LINK3D,
        input_dir_name    = CELLPOSE["warped_dir_name"],
        output_dir_name   = LINK3D["output_dir_name"],
        denoised_dir_name = DENOISE["output_dir_name"],
    log:
        f"{LOG_BASE}/link_3d_cells/{{core}}.log"
    threads: 2
    shell:
        "{PYTHON} {params.script} --core_name {wildcards.core} "
        "--min_slices {params.L[min_slices]} --max_slices {params.L[max_slices]} "
        "--min_area_px {params.L[min_area_px]} --min_overlap {params.L[min_overlap]} "
        "--min_confirmed {params.L[min_confirmed]} --min_iou {params.L[min_iou]} "
        "--min_intensity_frac {params.L[min_intensity_frac]} "
        "--coloc_radius_um {params.L[coloc_radius_um]} "
        "--input_dir_name {params.input_dir_name} "
        "--output_dir_name {params.output_dir_name} "
        "--denoised_dir_name {params.denoised_dir_name} --plot_qc > {log} 2>&1"


# =============================================================================
# STAGE 5 — PHENOTYPING
# =============================================================================
rule phenotype_cells:
    input:
        warped_dir = f"{DATASPACE}/{CELLPOSE['warped_dir_name']}/{{core}}",
        denoised   = f"{DATASPACE}/{DENOISE['output_dir_name']}/{{core}}/{{core}}_denoised.ome.tif",
    output:
        phenotypes = f"{DATASPACE}/{PHENO['output_dir_name']}/{{core}}/{{core}}_phenotypes.csv",
    params:
        script            = f"{SCRIPTS}/spatial_analysis/phenotype_cells.py",
        min_area_px       = PHENO["min_area_px"],
        denoised_dir_name = DENOISE["output_dir_name"],
        mask_dir_name     = CELLPOSE["warped_dir_name"],
        output_dir_name   = PHENO["output_dir_name"],
        reg_stats_flag    = (f"--reg_stats_csv {PHENO['reg_stats_csv']}"
                             if PHENO["reg_stats_csv"] else ""),
    log:
        f"{LOG_BASE}/phenotype_cells/{{core}}.log"
    threads: 2
    shell:
        "{PYTHON} {params.script} --core_name {wildcards.core} "
        "--min_area_px {params.min_area_px} "
        "--denoised_dir_name {params.denoised_dir_name} "
        "--mask_dir_name {params.mask_dir_name} "
        "--output_dir_name {params.output_dir_name} "
        "{params.reg_stats_flag} --plot_qc > {log} 2>&1"


# =============================================================================
# STAGE 6 — CELL TYPE ASSIGNMENT
# =============================================================================
rule assign_phenotypes:
    input:
        phenotypes = f"{DATASPACE}/{PHENO['output_dir_name']}/{{core}}/{{core}}_phenotypes.csv",
        map_       = f"{DATASPACE}/{LINK3D['output_dir_name']}/{{core}}/{{core}}_DAPI_2d_to_3d_map.csv",
        stats      = f"{DATASPACE}/{LINK3D['output_dir_name']}/{{core}}/{{core}}_DAPI_3d_stats.csv",
    output:
        typed_2d = f"{DATASPACE}/{PHENO['output_dir_name']}/{{core}}/{{core}}_phenotypes_typed.csv",
        typed_3d = f"{DATASPACE}/{PHENO['output_dir_name']}/{{core}}/{{core}}_3d_typed.csv",
    params:
        script             = f"{SCRIPTS}/spatial_analysis/assign_phenotypes.py",
        min_confidence     = ASSIGN["min_confidence"],
        phenotype_dir_name = PHENO["output_dir_name"],
        linking_dir_name   = LINK3D["output_dir_name"],
        output_dir_name    = PHENO["output_dir_name"],
    log:
        f"{LOG_BASE}/assign_phenotypes/{{core}}.log"
    threads: 1
    shell:
        "{PYTHON} {params.script} --core_name {wildcards.core} "
        "--min_confidence {params.min_confidence} "
        "--phenotype_dir_name {params.phenotype_dir_name} "
        "--linking_dir_name {params.linking_dir_name} "
        "--output_dir_name {params.output_dir_name} > {log} 2>&1"


# =============================================================================
# STAGE 7 — 2D vs 3D TME COMPARISON
# =============================================================================
rule compare_2d_3d_tme:
    input:
        typed_2d = f"{DATASPACE}/{PHENO['output_dir_name']}/{{core}}/{{core}}_phenotypes_typed.csv",
        typed_3d = f"{DATASPACE}/{PHENO['output_dir_name']}/{{core}}/{{core}}_3d_typed.csv",
    output:
        summary = f"{DATASPACE}/{TME['output_dir_name']}/{{core}}/summary_comparison.csv",
    params:
        script             = f"{SCRIPTS}/spatial_analysis/compare_2d_3d_tme.py",
        radius_um          = TME["radius_um"],
        min_cells          = TME["min_cells"],
        phenotype_dir_name = PHENO["output_dir_name"],
        output_dir_name    = TME["output_dir_name"],
    log:
        f"{LOG_BASE}/compare_2d_3d_tme/{{core}}.log"
    threads: 1
    shell:
        "{PYTHON} {params.script} --core_name {wildcards.core} "
        "--radius_um {params.radius_um} --min_cells {params.min_cells} "
        "--phenotype_dir_name {params.phenotype_dir_name} "
        "--output_dir_name {params.output_dir_name} > {log} 2>&1"


# =============================================================================
# STAGE 8 — AGGREGATION
# =============================================================================
rule aggregate_tme:
    input:
        expand(
            f"{DATASPACE}/{{tme_dir}}/{{core}}/summary_comparison.csv",
            tme_dir=[TME["output_dir_name"]], core=CORES,
        ),
    output:
        summary = f"{DATASPACE}/{TME['output_dir_name']}/Aggregate/aggregate_summary.csv",
    params:
        script       = f"{SCRIPTS}/spatial_analysis/aggregate_tme.py",
        radius_um    = AGG["radius_um"],
        tme_dir_name = TME["output_dir_name"],
    log:
        f"{LOG_BASE}/aggregate_tme/aggregate.log"
    threads: 1
    shell:
        "{PYTHON} {params.script} --radius_um {params.radius_um} "
        "--tme_dir_name {params.tme_dir_name} > {log} 2>&1"


# =============================================================================
# STAGE 9 — 3D CELL RENDERING (Always Runs)
# =============================================================================
rule render_3d_cells:
    input:
        labels = f"{DATASPACE}/{LINK3D['output_dir_name']}/{{core}}/{{core}}_DAPI_3d_labels.tif",
        stats  = f"{DATASPACE}/{LINK3D['output_dir_name']}/{{core}}/{{core}}_DAPI_3d_stats.csv",
    output:
        done = touch(f"{DATASPACE}/{LINK3D['output_dir_name']}/{{core}}/qc/render_3d_qc/.done"),
    params:
        script         = f"{SCRIPTS}/spatial_analysis/render_3d_cells.py",
        input_dir_name = LINK3D["output_dir_name"],
        min_confirmed  = RENDER3D["min_confirmed"],
        n_samples      = RENDER3D["n_samples"],
    log:
        f"{LOG_BASE}/render_3d_cells/{{core}}.log"
    threads: 1
    shell:
        "{PYTHON} {params.script} --core_name {wildcards.core} "
        "--input_dir_name {params.input_dir_name} "
        "--min_confirmed {params.min_confirmed} --n_samples {params.n_samples} > {log} 2>&1"