#!/usr/bin/env python3
"""
Compare Page Fault Metrics: H100 (SW Coherence) vs GH200 (HW Coherence)

Produces a single dual-axis bar chart:
  Left y-axis  (log): Page Fault Count amplification ratio (H100 / GH200)
  Right y-axis (log): H100 Coherence Migration in MB (HtoD + DtoH)

Bars are colored by workload partitioning group.
Configuration: threads=64, gpu_blocks=64, partition=0.5 (static) or dynamic.
"""

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')

SCRIPT_DIR = Path(__file__).parent
GH200_DIR  = SCRIPT_DIR / "gh200_pf"
H100_DIR   = SCRIPT_DIR / "h100_pf"
OUTPUT_DIR = SCRIPT_DIR / "comparison_plots_pf"

TARGET_THREADS    = 64
TARGET_GPU_BLOCKS = 64
TARGET_PARTITION_STATIC  = 0.5
TARGET_PARTITION_DYNAMIC = "dynamic"

DYNAMIC_WORKLOADS = {'SSSP', 'RSCT', 'TQ', 'TQH', 'TRNS', 'BFS', 'CEDT'}

WORKLOAD_GROUPS = {
    'Data Partitioning':        ['BS', 'CEDD', 'HSTI', 'HSTO', 'RSCD', 'PAD', 'SC'],
    'Fine Task Partitioning':   ['TQ', 'RSCT', 'TQH'],
    'Coarse Task Partitioning': ['BFS', 'CEDT', 'SSSP'],
}

EXCLUDED_WORKLOADS = set(['TRNS', 'HSTI', 'RSCD', 'TQH'])  # include all workloads

# Flat ordering for x-axis
WORKLOAD_ORDER = []
for _wl in WORKLOAD_GROUPS.values():
    WORKLOAD_ORDER.extend(_wl)

# Colors per partitioning group (used for x-axis label coloring only)
GROUP_COLORS = {
    'Data Partitioning':        '#A0C4FF',   # blue
    'Fine Task Partitioning':   '#FFD6A5',   # orange
    'Coarse Task Partitioning': '#FFB3BA',   # green
}

# Per-metric bar colors
METRIC_COLORS = {
    'pf':        '#1f77b4',   # blue
    'coh_total': '#ff7f0e',   # orange
}

METRIC_LABELS = {
    'pf':        'PF Count Ratio (H100 / GH200)',
    'coh_total': 'H100 Coherence Migration MB',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _workload_to_group(workload):
    for gname, glist in WORKLOAD_GROUPS.items():
        if workload in glist:
            return gname
    return None


def load_pagefault_results(results_dir):
    data = {}
    for csv_file in glob.glob(os.path.join(results_dir, "*_pagefaults.csv")):
        bench = os.path.basename(csv_file).replace("_pagefaults.csv", "")
        df = pd.read_csv(csv_file)
        if 'partition' in df.columns:
            df['partition_str'] = df['partition'].astype(str)
            df['partition_num'] = pd.to_numeric(df['partition'], errors='coerce')
        data[bench] = df
    return data


def filter_config(df, bench_name):
    is_dynamic = bench_name in DYNAMIC_WORKLOADS
    mask = (df['threads'] == TARGET_THREADS) & (df['gpu_blocks'] == TARGET_GPU_BLOCKS)
    if is_dynamic:
        mask &= (df['partition_str'] == TARGET_PARTITION_DYNAMIC)
    else:
        mask &= (df['partition_num'] == TARGET_PARTITION_STATIC)
    return df[mask]


def safe_ratio(baseline, comparison):
    if baseline == 0:
        return 1.0 if comparison == 0 else float('inf')
    return comparison / baseline


def compute_metrics(gh200_data, h100_data):
    rows = []
    common = (set(gh200_data) & set(h100_data)) - EXCLUDED_WORKLOADS
    for wl in sorted(common):
        g = filter_config(gh200_data[wl], wl)
        h = filter_config(h100_data[wl], wl)
        if g.empty or h.empty:
            print(f"  Warning: no matching config for {wl}")
            continue
        gr, hr = g.iloc[0], h.iloc[0]

        gh200_pf = gr['cpu_page_faults'] + gr['gpu_page_faults']
        h100_pf  = hr['cpu_page_faults'] + hr['gpu_page_faults']

        gh200_htod_coh = gr.get('htod_coherence_mb', 0) or 0
        h100_htod_coh  = hr.get('htod_coherence_mb', 0) or 0
        gh200_dtoh_coh = gr.get('dtoh_coherence_mb', 0) or 0
        h100_dtoh_coh  = hr.get('dtoh_coherence_mb', 0) or 0

        # Combined coherence migration (HtoD + DtoH)
        gh200_coh_total = gh200_htod_coh + gh200_dtoh_coh
        h100_coh_total  = h100_htod_coh  + h100_dtoh_coh

        rows.append({
            'workload':       wl,
            'pf_ratio':       safe_ratio(gh200_pf, h100_pf),
            'coh_total_ratio': safe_ratio(gh200_coh_total, h100_coh_total),
            'htod_coh_ratio': safe_ratio(gh200_htod_coh, h100_htod_coh),
            'dtoh_coh_ratio': safe_ratio(gh200_dtoh_coh, h100_dtoh_coh),
            'gh200_pf':       gh200_pf,
            'h100_pf':        h100_pf,
            'gh200_coh_total': gh200_coh_total,
            'h100_coh_total':  h100_coh_total,
            'gh200_htod_coh': gh200_htod_coh,
            'h100_htod_coh':  h100_htod_coh,
            'gh200_dtoh_coh': gh200_dtoh_coh,
            'h100_dtoh_coh':  h100_dtoh_coh,
            'is_dynamic':     wl in DYNAMIC_WORKLOADS,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(metrics_df, output_path):
    """Single-axis bar chart: PF count ratio (H100 / GH200), log y-axis."""

    df = metrics_df.copy()
    df['sort_order'] = df['workload'].apply(
        lambda w: WORKLOAD_ORDER.index(w) if w in WORKLOAD_ORDER else 999)
    df = df.sort_values('sort_order').reset_index(drop=True)

    workloads = df['workload'].values
    n         = len(workloads)
    bar_width = 0.6
    x         = np.arange(n)
    pf_vals   = df['pf_ratio'].values.copy()

    # Assign bar colour by partitioning group
    bar_colors = [GROUP_COLORS.get(_workload_to_group(w), '#888888')
                  for w in workloads]

    fig, ax = plt.subplots(figsize=(22, 10))

    bars = ax.bar(x, pf_vals, bar_width,
                  color=bar_colors, edgecolor='black', linewidth=0.6, zorder=3)

    # --- value labels --------------------------------------------------------
    for bar, v in zip(bars, pf_vals):
        label = '\u221e' if np.isinf(v) else f'{v:.1f}\u00d7'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2,
                        bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=22, fontweight='bold', rotation=90)

    # --- reference line at 1x ------------------------------------------------
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2.5, alpha=0.7)

    # --- log scale ------------------------------------------------------------
    ax.set_yscale('log')
    pf_finite = [v for v in pf_vals if np.isfinite(v) and v > 0]
    if pf_finite:
        ax.set_ylim(bottom=min(pf_finite) / 2, top=max(pf_finite) * 5)

    # --- group separator lines -----------------------------------------------
    cum = 0
    boundaries = []
    for gname, glist in WORKLOAD_GROUPS.items():
        cnt = sum(1 for w in glist if w in workloads)
        if cnt:
            cum += cnt
            boundaries.append((cum - 0.5, gname, cnt))

    for i, (bx, _, _) in enumerate(boundaries[:-1]):
        ax.axvline(x=bx, color='black', linewidth=1.5, alpha=0.5)

    # --- axes ----------------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha='right', fontsize=22,
                       color='black')

    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel('Workload', fontsize=26, fontweight='bold')
    ax.set_ylabel('Page-Fault Count Ratio  (H100 / GH200)',
                  fontsize=24, fontweight='bold')
    ax.set_title(
        'Page-Fault Amplification:  x86 + H100 (SW)  vs  GH200 (HW)\n'
        f'threads={TARGET_THREADS}, gpu_blocks={TARGET_GPU_BLOCKS}, '
        f'partition={TARGET_PARTITION_STATIC} (static) / dynamic',
        fontsize=24, fontweight='bold')

    # --- legend (one patch per group + reference line) -----------------------
    group_handles = [Patch(facecolor=GROUP_COLORS[g], edgecolor='black',
                           label=g)
                     for g in WORKLOAD_GROUPS if g in GROUP_COLORS]
    ref_line = plt.Line2D([0], [0], color='gray', linestyle='--',
                          linewidth=2.5, label='Equal (1\u00d7)')
    ax.legend(handles=group_handles + [ref_line],
              loc='upper left', fontsize=18, framealpha=0.9)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(df):
    print("\n" + "=" * 130)
    print("PAGE FAULT COMPARISON  —  H100 (SW Coherence) vs GH200 (HW Coherence)")
    print(f"Config: threads={TARGET_THREADS}, gpu_blocks={TARGET_GPU_BLOCKS}, "
          f"partition={TARGET_PARTITION_STATIC} (static) / dynamic")
    print("=" * 130)
    hdr = (f"{'Workload':<8} {'Type':<8} "
           f"{'PF Amp':>10} {'Coh Mig':>10}   "
           f"{'GH200 PF':>10} {'H100 PF':>10}   "
           f"{'GH200 Coh':>10} {'H100 Coh':>10}")
    print(hdr)
    print("-" * 100)
    for _, r in df.iterrows():
        tp = "Dyn" if r['is_dynamic'] else "Stat"
        print(f"{r['workload']:<8} {tp:<8} "
              f"{r['pf_ratio']:>9.2f}x {r['coh_total_ratio']:>9.2f}x   "
              f"{r['gh200_pf']:>10.0f} {r['h100_pf']:>10.0f}   "
              f"{r['gh200_coh_total']:>10.2f} {r['h100_coh_total']:>10.2f}")
    print("=" * 100)
    print("Ratio > 1× → H100 has more  |  Ratio < 1× → GH200 has more\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading GH200 data from {GH200_DIR} …")
    gh200 = load_pagefault_results(GH200_DIR)
    print(f"  {len(gh200)} workloads")

    print(f"Loading H100  data from {H100_DIR} …")
    h100 = load_pagefault_results(H100_DIR)
    print(f"  {len(h100)} workloads")

    metrics = compute_metrics(gh200, h100)
    if metrics.empty:
        print("ERROR: no matching workloads!")
        return

    print_summary(metrics)

    out = OUTPUT_DIR / "pf_comparison.png"
    plot_comparison(metrics, out)

    csv_out = OUTPUT_DIR / "pf_comparison_metrics.csv"
    metrics.to_csv(csv_out, index=False)
    print(f"Saved CSV:  {csv_out}")


if __name__ == "__main__":
    main()
