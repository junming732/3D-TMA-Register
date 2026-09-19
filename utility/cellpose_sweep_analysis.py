"""Analyzes and visualizes CellPose parameter sweep results from a summary CSV.

Reads the summary CSV output from a CellPose parameter sweep and generates 
multi-panel line plots to evaluate the impact of `flow_threshold` and 
`cellprob_threshold` on segmentation performance. The module evaluates both 
the total number of detected cells and the mean physical diameter of the 
resulting masks. Pixel-to-micrometer conversion is applied to contextualize 
mask sizes against typical mammalian biological parameters.

Example:
    Standard execution from the command line (requires updating the `TARGET_CSV` 
    variable in the script entry point):
    $ python cellpose_sweep_analysis.py

Notes:
    The script expects the input CSV to contain `flow_threshold`, `n_cells`, 
    `cellprob_threshold`, `diameter_px`, and `mean_diameter_px` columns.

    Generates two high-resolution (300 DPI) output visualizations in the current 
    working directory:
    1. `sweep_summary_n_cells_tworows.png`: Visualizes cell detection counts.
    2. `sweep_summary_mean_diam_um_tworows.png`: Visualizes mean mask dimensions 
       with a highlighted biological reference band (5-10 µm) for validation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_sweep_summary_plots(csv_path: str, pixel_size_um: float = 0.4961):
    """
    Reads a CellPose parameter sweep summary CSV and generates two-row line plots 
    visualizing the impact of flow and cellprob thresholds on segmentation.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return

    # Load data
    df = pd.read_csv(csv_path)

    # Convert pixel diameters to physical units (micrometers)
    if 'mean_diameter_px' in df.columns:
        df['mean_diameter_um'] = df['mean_diameter_px'] * pixel_size_um

    # Set visualization style suitable for academic figures
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # ---------------------------------------------------------
    # Plot 1: Impact on Raw Cell Count
    # ---------------------------------------------------------
    g1 = sns.relplot(
        data=df, 
        x="flow_threshold", 
        y="n_cells", 
        hue="cellprob_threshold",
        col="diameter_px", 
        col_wrap=3,  
        kind="line", 
        marker="o", 
        palette="viridis", 
        height=4, 
        aspect=1.2,
        linewidth=2.5
    )
    g1.set_axis_labels("Flow Threshold (Shape Tolerance)", "Number of Cells Detected")
    g1.set_titles("Base Diameter: {col_name}")
    g1.legend.set_title("Cellprob\nThreshold")
    
    g1.fig.suptitle("Effect of Parameter Tuning on Total Cancer Cell Detection", fontweight='bold')
    plt.subplots_adjust(top=0.9)
    
    out_path_cells = "sweep_summary_n_cells_tworows.png"
    plt.savefig(out_path_cells, dpi=300, bbox_inches='tight')
    print(f"Saved cell count plot -> {out_path_cells}")
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Impact on Spatial Mask Size (Physical Dimensions)
    # ---------------------------------------------------------
    g2 = sns.relplot(
        data=df, 
        x="flow_threshold", 
        y="mean_diameter_um", 
        hue="cellprob_threshold",
        col="diameter_px", 
        col_wrap=3,  
        kind="line", 
        marker="o", 
        palette="viridis",  # Standardized to match Plot 1
        height=4, 
        aspect=1.2,
        linewidth=2.5
    )
    
    # Add a reference band for typical mammalian nucleus size (5 - 10 µm)
    for ax in g2.axes.flat:
        ax.axhspan(5, 10, color='gray', alpha=0.2, zorder=0, label='Normal Nucleus Range (5-10 µm)')
    
    g2.set_axis_labels("Flow Threshold (Shape Tolerance)", "Mean Mask Diameter (µm)")
    g2.set_titles("Base Diameter: {col_name}")
    g2.legend.set_title("Cellprob\nThreshold")
    
    g2.fig.suptitle("Effect of Parameter Tuning on Physical Mask Boundaries", fontweight='bold')
    plt.subplots_adjust(top=0.9)
    
    out_path_size = "sweep_summary_mean_diam_um_tworows.png"
    plt.savefig(out_path_size, dpi=300, bbox_inches='tight')
    print(f"Saved mask size plot -> {out_path_size}")
    plt.close()

if __name__ == "__main__":
    TARGET_CSV = "xxx.csv"
    generate_sweep_summary_plots(TARGET_CSV)