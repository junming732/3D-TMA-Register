"""
summarize_landmarks.py
=======================
Produces a small summary table of a landmark annotation JSON file

Auto-detects two possible JSON schemas:
  (a) Flat list:      [{"x":.., "y":.., "z":.., "landmark_id":..}, ...]
  (b) Nested:          {"annotations": [{"z":.., "mclass":.., "id":..,
                                          "points": [{"x":.., "y":..}]}, ...]}

Outputs a plain-text table and a ready-to-paste LaTeX table (booktabs style,
matching the rest of the thesis).

Usage
-----
    python summarize_landmarks.py --json /path/to/landmark_annotation_core_09.json \
                                   --core_name "Core 09"
"""

import json
import argparse
from collections import defaultdict, Counter


def load_structures(json_path):
    """Return dict: structure_id -> list of {'x','y','z'} dicts, regardless of schema."""
    with open(json_path) as f:
        data = json.load(f)

    by_struct = defaultdict(list)

    if isinstance(data, list):
        # Flat schema: [{x, y, z, landmark_id}, ...]
        for d in data:
            sid = d['landmark_id']
            by_struct[sid].append({'x': d['x'], 'y': d['y'], 'z': d['z']})
    elif isinstance(data, dict) and 'annotations' in data:
        # Nested schema: {"annotations": [{z, mclass, id, points:[{x,y}]}]}
        for ann in data['annotations']:
            sid = ann['mclass']
            pt = ann['points'][0]
            by_struct[sid].append({'x': pt['x'], 'y': pt['y'], 'z': ann['z']})
    else:
        raise ValueError(f"Unrecognised JSON schema in {json_path}")

    return by_struct


def summarize(by_struct):
    counts = [len(v) for v in by_struct.values()]
    n_structures = len(by_struct)
    n_points = sum(counts)

    z_all = [pt['z'] for pts in by_struct.values() for pt in pts]
    z_min, z_max = min(z_all), max(z_all)

    # Count valid consecutive (z, z+1) pairs per structure (same definition as TRE eval)
    n_pairs = 0
    structures_with_pairs = 0
    for sid, pts in by_struct.items():
        zs = sorted(p['z'] for p in pts)
        zset = set(zs)
        has_pair = False
        for z in zs:
            if (z + 1) in zset:
                n_pairs += 1
                has_pair = True
        if has_pair:
            structures_with_pairs += 1

    singleton_structures = sum(1 for c in counts if c == 1)
    dist = Counter(counts)

    return {
        'n_structures': n_structures,
        'n_points': n_points,
        'n_pairs': n_pairs,
        'structures_with_pairs': structures_with_pairs,
        'singleton_structures': singleton_structures,
        'z_min': z_min,
        'z_max': z_max,
        'min_pts': min(counts),
        'max_pts': max(counts),
        'mean_pts': n_points / n_structures,
        'distribution': dict(sorted(dist.items())),
    }


def print_text_table(stats, core_name):
    print(f"\nLandmark annotation summary — {core_name}")
    print("-" * 50)
    print(f"{'Total annotated points':<35}{stats['n_points']:>10}")
    print(f"{'Distinct anatomical structures':<35}{stats['n_structures']:>10}")
    print(f"{'Structures with >=1 valid TRE pair':<35}{stats['structures_with_pairs']:>10}")
    print(f"{'Singleton structures (no pair)':<35}{stats['singleton_structures']:>10}")
    print(f"{'Valid consecutive (z, z+1) pairs':<35}{stats['n_pairs']:>10}")
    print(f"{'Points per structure (min/mean/max)':<35}"
          f"{stats['min_pts']}/{stats['mean_pts']:.2f}/{stats['max_pts']:>5}")
    print(f"{'Annotated z-range':<35}{stats['z_min']}-{stats['z_max']:>6}")
    print(f"{'Points-per-structure distribution':<35}{stats['distribution']}")


def print_latex_table(stats, core_name):
    print(r"\begin{table}[ht]")
    print(r"    \centering")
    print(r"    \begin{tabular}{lr}")
    print(r"        \toprule")
    print(r"        \textbf{Quantity} & \textbf{Value} \\")
    print(r"        \midrule")
    print(f"        Total annotated points & {stats['n_points']} \\\\")
    print(f"        Distinct anatomical structures & {stats['n_structures']} \\\\")
    print(f"        Structures with at least one valid pair & "
          f"{stats['structures_with_pairs']} \\\\")
    print(f"        Valid consecutive landmark pairs ($K$) & {stats['n_pairs']} \\\\")
    print(f"        Points per structure (min / mean / max) & "
          f"{stats['min_pts']} / {stats['mean_pts']:.2f} / {stats['max_pts']} \\\\")
    print(f"        Annotated $z$-range & {stats['z_min']}--{stats['z_max']} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(f"    \\caption{{Summary of landmark annotations for {core_name}, "
          f"used for the registration evaluation in Section~9.1.1.}}")
    print(r"    \label{tab:landmark_summary}")
    print(r"\end{table}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--core_name', default='Core 09')
    args = ap.parse_args()

    by_struct = load_structures(args.json)
    stats = summarize(by_struct)
    print_text_table(stats, args.core_name)
    print_latex_table(stats, args.core_name)