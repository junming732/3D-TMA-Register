"""
analyze_registration_channels.py
=================================

Standalone analysis of the row-level output from analyse_unsuitable_cores.py.

Takes a single, combined *_akaze_summary.csv (all cores run together) and
answers three questions:

  1. EVIDENCE   — which channel actually has the highest AKAZE success rate,
                  compared across the unsuitable group and the other group?
                  -> evidence_channel_success.png

  2. PER-CORE   — for each unsuitable core, which channel wins on which
                  slice-pair? -> one heatmap PNG per unsuitable core,
                  channels x slice-pairs, colored by success/inliers.

  3. RESCUE     — for each unsuitable core, every slice-pair where CK (or
                  whatever --primary-channel you pick) failed but some other
                  channel still succeeded. -> rescue_candidates.csv

Usage:
    python analyze_registration_channels.py \
        --input "all_cores_summary_by_channel(dapi_clahe).csv" \
        --out-dir ./registration_analysis \
        --primary-channel CK

--input only needs a filename: the script looks it up automatically inside
config.DATASPACE/Unsuitable_Core_Diagnostics (the same folder the upstream
script writes it to). Pass a path containing "/" (or an absolute path) to
point somewhere else instead.

Which cores count as "Unsuitable" (difficult) vs "Other" (easy) is no longer
inferred from which file a row came from — since everything is in one CSV
now — but from the DIFFICULT_CORES list below. Edit that list to match
whichever cores you currently consider flagged/difficult, or override it
per-run with --difficult-cores (comma-separated core names).

All three outputs land in --out-dir. Nothing here needs the original TIFFs
or masks — it only reads the summary CSV.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    import config
    CSV_DIR = os.path.join(config.DATASPACE, "Unsuitable_Core_Diagnostics")
except ImportError:
    # config.py isn't importable from wherever this script is being run from
    # (e.g. different working directory / not on PYTHONPATH). --input will
    # just be used as-is (relative or absolute path) in that case.
    config = None
    CSV_DIR = None



# ─────────────────────────────────────────────────────────────────────────
# EASY / DIFFICULT CORE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────
# Edit this list to whatever cores you currently consider flagged/"difficult"
# (previously the ones that lived in the separate unsuitable_cores CSV).
# Every other Core found in the combined CSV is treated as "Other" (easy).
# Names must match the values in the CSV's "Core" column exactly.
# Can also be overridden per-run with --difficult-cores.
DIFFICULT_CORES = [
    "Core_16",
    "Core_17",
    "Core_21",
    "Core_23",
    "Core_27",
]
# Everything else in the range 1-30 (core_01, core_02, ... core_30, minus
# the ones above) is automatically treated as "Other"/normal — you don't
# need to list them, that's handled by the isin() check below.


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Analyze registration-channel diagnostics.")
parser.add_argument("--input", default="all_cores_summary_by_channel(dapi_clahe).csv",
                     help="CSV filename (just the filename, unless you want to point "
                          "somewhere else). Looked up inside "
                          f"{CSV_DIR!r} (config.DATASPACE/Unsuitable_Core_Diagnostics) "
                          "unless you pass an absolute path or a path containing '/'.")
parser.add_argument("--out-dir", default=None,
                     help="Directory to write PNGs/CSVs into. Defaults to a "
                          "'registration_analysis' subfolder inside the same "
                          "directory the input CSV was found in (CSV_DIR).")
parser.add_argument("--primary-channel", default="CK",
                     help="Channel to treat as the default/primary choice for the rescue analysis.")
parser.add_argument("--min-inliers-trust", type=int, default=15,
                     help="Inlier count below which a 'success' is flagged as low-confidence "
                          "in the rescue table (informational only, doesn't filter rows).")
parser.add_argument("--difficult-cores", default=None,
                     help="Comma-separated list of Core names to treat as 'Unsuitable'/difficult, "
                          "overriding the DIFFICULT_CORES list defined at the top of this script.")
args = parser.parse_args()

difficult_cores = (
    [c.strip() for c in args.difficult_cores.split(",") if c.strip()]
    if args.difficult_cores is not None
    else DIFFICULT_CORES
)


# ─────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────
def resolve_input_path(name):
    # Absolute path or one that already includes a directory -> use as-is.
    if os.path.isabs(name) or os.path.dirname(name):
        return name
    if CSV_DIR is None:
        raise RuntimeError(
            "Couldn't import 'config' to locate Unsuitable_Core_Diagnostics, and "
            f"'{name}' has no directory in it. Either run this script from "
            "somewhere config.py is importable, or pass a full/relative path via --input."
        )
    return os.path.join(CSV_DIR, name)


input_path = resolve_input_path(args.input)

out_dir = args.out_dir
if out_dir is None:
    out_dir = os.path.join(os.path.dirname(input_path) or ".", "registration_analysis")
args.out_dir = out_dir
os.makedirs(args.out_dir, exist_ok=True)

df = pd.read_csv(input_path)
df["Group"] = np.where(df["Core"].isin(difficult_cores), "Unsuitable", "Other")

unknown = set(difficult_cores) - set(df["Core"].unique())
if unknown:
    print(f"[warning] These difficult_cores entries don't match any Core in the CSV: "
          f"{sorted(unknown)}")

# Slice-pair ordering: "Z00_ID1_to_Z01_ID2" -> 0
def z_order(pair_str):
    m = re.match(r"Z(\d+)_", pair_str)
    return int(m.group(1)) if m else -1

df["ZOrder"] = df["Slice_Pair"].apply(z_order)

CHANNELS = sorted(df["Channel"].unique())
if args.primary_channel not in CHANNELS:
    raise ValueError(f"--primary-channel '{args.primary_channel}' not found. "
                      f"Available: {CHANNELS}")

merged_path = os.path.join(args.out_dir, "merged_akaze_summary.csv")
df.to_csv(merged_path, index=False)
n_unsuitable = df.loc[df["Group"] == "Unsuitable", "Core"].nunique()
n_other = df.loc[df["Group"] == "Other", "Core"].nunique()
print(f"[load] Read {input_path}")
print(f"[merge] {len(df)} rows ({df['Core'].nunique()} cores, "
      f"{n_unsuitable} unsuitable / {n_other} other) -> {merged_path}")


# ─────────────────────────────────────────────────────────────────────────
# 1. EVIDENCE — success rate per channel, per group
#    (computed per-core first, then averaged, so a core with more/fewer
#    pairs never silently dominates the mean)
# ─────────────────────────────────────────────────────────────────────────
def per_core_success_rate(sub_df):
    return sub_df.groupby(["Core", "Channel"])["AKAZE_OK"].mean().mul(100).reset_index()

rates = per_core_success_rate(df)
rates = rates.merge(df[["Core", "Group"]].drop_duplicates(), on="Core")
evidence = rates.groupby(["Channel", "Group"])["AKAZE_OK"].mean().unstack("Group")
evidence = evidence.sort_values("Other", ascending=False)

print("\n[evidence] Mean per-core success rate (%) by channel:")
print(evidence.round(1).to_string())

fig, ax = plt.subplots(figsize=(8, 0.55 * len(evidence) + 1.5))
y = np.arange(len(evidence))
bar_h = 0.36
ax.barh(y + bar_h/2, evidence["Other"], height=bar_h, color="#4d7fff", label="Other cores")
ax.barh(y - bar_h/2, evidence["Unsuitable"], height=bar_h, color="#ffb020", label="Unsuitable cores")
ax.set_yticks(y)
ax.set_yticklabels(evidence.index)
ax.set_xlabel("Mean per-core AKAZE success rate (%)")
ax.set_xlim(0, 100)
ax.invert_yaxis()
ax.legend(loc="lower right", frameon=False)
ax.set_title("Channel success rate: unsuitable cores vs. other cores")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
evidence_path = os.path.join(args.out_dir, "evidence_channel_success.png")
fig.savefig(evidence_path, dpi=150)
plt.close(fig)
print(f"[evidence] Saved -> {evidence_path}")


# ─────────────────────────────────────────────────────────────────────────
# 2. PER-CORE HEATMAPS — channel x slice-pair, for each unsuitable core
# ─────────────────────────────────────────────────────────────────────────
def core_heatmap(core_name, core_df, out_path, primary_channel):
    pairs = (core_df[["Slice_Pair", "ZOrder"]]
             .drop_duplicates()
             .sort_values("ZOrder")["Slice_Pair"].tolist())

    # rank channels by this core's own success rate, best first
    ch_rate = core_df.groupby("Channel")["AKAZE_OK"].mean().sort_values(ascending=False)
    channels_ranked = ch_rate.index.tolist()

    ok_grid = np.zeros((len(channels_ranked), len(pairs)))
    inlier_grid = np.zeros_like(ok_grid)
    for i, ch in enumerate(channels_ranked):
        for j, p in enumerate(pairs):
            row = core_df[(core_df.Channel == ch) & (core_df.Slice_Pair == p)]
            if len(row):
                ok_grid[i, j] = 1 if row.iloc[0]["AKAZE_OK"] else 0
                inlier_grid[i, j] = row.iloc[0]["N_Inliers"]

    # color: red->green diverging by success, intensity by inlier count
    fail_cmap = LinearSegmentedColormap.from_list("fail", ["#2a1012", "#e5484d"])
    ok_cmap   = LinearSegmentedColormap.from_list("ok",   ["#0f2a1a", "#33d17a"])
    max_inlier = max(inlier_grid.max(), 1)

    fig, ax = plt.subplots(figsize=(0.5 * len(pairs) + 2, 0.4 * len(channels_ranked) + 1.5))
    rgb = np.zeros((*ok_grid.shape, 3))
    for i in range(ok_grid.shape[0]):
        for j in range(ok_grid.shape[1]):
            t = min(inlier_grid[i, j] / max_inlier, 1.0)
            cmap = ok_cmap if ok_grid[i, j] else fail_cmap
            rgb[i, j] = cmap(0.25 + 0.75 * t)[:3]
    ax.imshow(rgb, aspect="auto")

    # mark rescue cells: primary channel failed here but this channel succeeded
    primary_idx = channels_ranked.index(primary_channel) if primary_channel in channels_ranked else None
    if primary_idx is not None:
        for j in range(len(pairs)):
            if ok_grid[primary_idx, j] == 1:
                continue
            for i in range(len(channels_ranked)):
                if i != primary_idx and ok_grid[i, j] == 1:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                fill=False, edgecolor="#ffb020", linewidth=2))

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"Z{k:02d}" for k in range(len(pairs))], fontsize=8, rotation=0)
    ax.set_yticks(range(len(channels_ranked)))
    ax.set_yticklabels([f"{ch} ({ch_rate[ch]*100:.0f}%)" for ch in channels_ranked], fontsize=8)
    ax.set_title(f"{core_name} — channel x slice-pair "
                 f"(green=success, red=fail, amber outline=rescues {primary_channel})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


heatmap_dir = os.path.join(args.out_dir, "heatmaps")
os.makedirs(heatmap_dir, exist_ok=True)
for core_name, core_df in df[df.Group == "Unsuitable"].groupby("Core"):
    out_path = os.path.join(heatmap_dir, f"{core_name}_heatmap.png")
    core_heatmap(core_name, core_df, out_path, args.primary_channel)
    print(f"[heatmap] {core_name} -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────
# 3. RESCUE CANDIDATES — primary channel fails, another channel succeeds
# ─────────────────────────────────────────────────────────────────────────
rescue_rows = []
for (core, pair), g in df[df.Group == "Unsuitable"].groupby(["Core", "Slice_Pair"]):
    primary_row = g[g.Channel == args.primary_channel]
    if primary_row.empty or bool(primary_row.iloc[0]["AKAZE_OK"]):
        continue  # primary channel worked (or is missing) here, nothing to rescue

    alternatives = g[(g.Channel != args.primary_channel) & (g.AKAZE_OK)]
    if alternatives.empty:
        continue  # nothing rescued this pair either

    best_alt = alternatives.sort_values("N_Inliers", ascending=False).iloc[0]
    rescue_rows.append(dict(
        Core=core,
        Slice_Pair=pair,
        ZOrder=z_order(pair),
        Primary_Channel=args.primary_channel,
        Rescue_Channel=best_alt["Channel"],
        Rescue_Inliers=int(best_alt["N_Inliers"]),
        Rescue_Matches=int(best_alt["N_Matches"]),
        Low_Confidence=bool(best_alt["N_Inliers"] < args.min_inliers_trust),
    ))

rescue_df = pd.DataFrame(rescue_rows).sort_values(["Core", "ZOrder"])
rescue_path = os.path.join(args.out_dir, "rescue_candidates.csv")
rescue_df.to_csv(rescue_path, index=False)

print(f"\n[rescue] {len(rescue_df)} slice-pairs where {args.primary_channel} failed "
      f"but another channel succeeded -> {rescue_path}")
if len(rescue_df):
    print(rescue_df.to_string(index=False))
    n_low_conf = rescue_df["Low_Confidence"].sum()
    if n_low_conf:
        print(f"\n  Note: {n_low_conf} of these rescues have < {args.min_inliers_trust} inliers "
              f"— geometrically 'successful' per AKAZE_OK, but worth a manual look before trusting "
              f"the transform.")
else:
    print(f"  None found — every pair {args.primary_channel} missed was also unrecoverable "
          f"on every other candidate.")

print(f"\nDone. All outputs in: {os.path.abspath(args.out_dir)}")