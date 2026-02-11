`cropping_cores.py`: The primary ETL pipeline that detects, rotation-corrects, and extracts individual tissue cores from whole-slide TMA images into separate OME-TIFF files.

`generate_maps.py`: A diagnostic tool that generates annotated JPEG maps to visualize detected cores, grid alignment, and IDs for rapid inspection.

`visual_GIFs.py`: A visualization utility that compiles sequential core images into animated GIFs, allowing for rapid assessment of 3D alignment stability across slices.

`extract_thumbnails.py`: Extracts the slide thumbnail image from OME-TIFFs and applies CLAHE-based contrast enhancement to improve readability of barcodes and text.

`VALIS_register_core.py`: An automated execution engine that wraps the VALIS library to handle the full registration lifecycle for histological Z-stacks, including rigid and non-rigid alignment, topology safety checks, and OME-TIFF output generation.

`evaluate_batch.py`: A comprehensive validation suite designed to audit VALIS-registered datasets against strict Bioinformatics (2018) standards, utilizing TRE, Jaccard, NCC, and GLCM Smoothness metrics to generate detailed stability analytics and comparative quality rankings.