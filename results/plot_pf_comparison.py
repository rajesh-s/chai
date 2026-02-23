#!/usr/bin/env python3
"""
Compare Page Fault Metrics: H100 (SW Coherence) vs GH200 (HW Coherence)

Creates a grouped bar chart showing the percentage increase/decrease in:
1. Total Page Fault Count
2. HtoD Migration (MB)
3. DtoH Migration (MB)

for H100 (software coherence) relative to GH200 (hardware coherence baseline).

Configuration filter:
- threads=64, gpu_blocks=64
- partition=0.5 for static workloads, dynamic for task-based workloads
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')

# Results directories
SCRIPT_DIR = Path(__file__).parent
GH200_DIR = SCRIPT_DIR / "gh200_pf"
H100_DIR = SCRIPT_DIR / "h100_pf"
OUTPUT_DIR = SCRIPT_DIR / "comparison_plots_pf"

# Configuration filters - Primary config
TARGET_THREADS = 64
TARGET_GPU_BLOCKS = 64
TARGET_PARTITION_STATIC = 0.5  # For static partitioning workloads
TARGET_PARTITION_DYNAMIC = "dynamic"  # For task-based workloads

# Configuration filters - Secondary config (smaller scale)
TARGET_THREADS_2 = 4
TARGET_GPU_BLOCKS_2 = 8

# Task-based (dynamic) workloads vs static partitioning workloads
DYNAMIC_WORKLOADS = {'SSSP', 'RSCT', 'TQ', 'TQH', 'TRNS', 'BFS', 'CEDT'}
STATIC_WORKLOADS = {'BS', 'PAD', 'RSCD', 'SC', 'HSTI', 'HSTO', 'CEDD'}

# Workload grouping by partitioning type
WORKLOAD_GROUPS = {
    'Data Partitioning': ['BS', 'CEDD', 'HSTI', 'HSTO', 'RSCD', 'PAD', 'SC'],
    'Fine Task Partitioning': ['RSCT', 'TQ', 'TQH'],
    'Coarse Task Partitioning': ['BFS', 'CEDT', 'SSSP'],
}

# Workloads to exclude from plot
EXCLUDED_WORKLOADS = {'TRNS', 'TQH', 'BS', 'HSTI', 'HSTO', 'RSCD'}

# Flattened order for plotting
WORKLOAD_ORDER = []
for group_workloads in WORKLOAD_GROUPS.values():
    WORKLOAD_ORDER.extend(group_workloads)

# Bar colors and patterns
COLORS = {
    'pf_count': '#1f77b4',    # Blue for page fault count
    'htod_mig': '#ff7f0e',    # Orange for HtoD migration
    'dtoh_mig': '#2ca02c',    # Green for DtoH migration
}

HATCHES = {
    'pf_count': '/',
    'htod_mig': '\\',
    'dtoh_mig': 'x',
}

# Colors per workload group for comparison plot
GROUP_COLORS = {
    'Data Partitioning': '#1f77b4',       # Blue
    'Fine Task Partitioning': '#ff7f0e',  # Orange
    'Coarse Task Partitioning': '#2ca02c', # Green
}

def _workload_to_group(workload):
    """Return the group name for a given workload."""
    for group_name, group_workloads in WORKLOAD_GROUPS.items():
        if workload in group_workloads:
            return group_name
    return None


def load_pagefault_results(results_dir):
    """Load all *_pagefaults.csv files from a directory."""
    data = {}
    csv_files = glob.glob(os.path.join(results_dir, "*_pagefaults.csv"))
    
    for csv_file in csv_files:
        bench_name = os.path.basename(csv_file).replace("_pagefaults.csv", "")
        df = pd.read_csv(csv_file)
        
        # Handle partition column - keep 'dynamic' as string, convert numbers
        if 'partition' in df.columns:
            # Create a copy for numeric comparison
            df['partition_str'] = df['partition'].astype(str)
            df['partition_num'] = pd.to_numeric(df['partition'], errors='coerce')
        
        data[bench_name] = df
    
    return data


def filter_config(df, bench_name, threads=TARGET_THREADS, gpu_blocks=TARGET_GPU_BLOCKS):
    """Filter dataframe for the target configuration."""
    # Determine if this is a dynamic or static workload
    is_dynamic = bench_name in DYNAMIC_WORKLOADS
    
    # Filter by threads and gpu_blocks
    mask = (df['threads'] == threads) & (df['gpu_blocks'] == gpu_blocks)
    
    # Filter by partition
    if is_dynamic:
        mask &= (df['partition_str'] == TARGET_PARTITION_DYNAMIC)
    else:
        mask &= (df['partition_num'] == TARGET_PARTITION_STATIC)
    
    filtered = df[mask]
    return filtered


def compute_ratio(baseline_val, comparison_val):
    """Compute ratio of comparison to baseline.
    
    Returns: comparison / baseline
    >1 = H100 has more, <1 = GH200 has more
    """
    if baseline_val == 0:
        if comparison_val == 0:
            return 1.0
        return float('inf') if comparison_val > 0 else float('-inf')
    return comparison_val / baseline_val


def compute_metrics(gh200_data, h100_data, threads=TARGET_THREADS, gpu_blocks=TARGET_GPU_BLOCKS):
    """Compute percentage change metrics for all workloads."""
    results = []
    
    # Find common workloads (excluding excluded ones)
    common_workloads = (set(gh200_data.keys()) & set(h100_data.keys())) - EXCLUDED_WORKLOADS
    
    for workload in sorted(common_workloads):
        gh200_df = gh200_data[workload]
        h100_df = h100_data[workload]
        
        # Filter for target configuration
        gh200_filtered = filter_config(gh200_df, workload, threads, gpu_blocks)
        h100_filtered = filter_config(h100_df, workload, threads, gpu_blocks)
        
        if gh200_filtered.empty or h100_filtered.empty:
            print(f"Warning: No data for {workload} with target config")
            continue
        
        # Use first matching row (should be unique with our filters)
        gh200_row = gh200_filtered.iloc[0]
        h100_row = h100_filtered.iloc[0]
        
        # Compute total page faults
        gh200_total_pf = gh200_row['cpu_page_faults'] + gh200_row['gpu_page_faults']
        h100_total_pf = h100_row['cpu_page_faults'] + h100_row['gpu_page_faults']
        
        # Compute ratios (H100 / GH200)
        pf_count_ratio = compute_ratio(gh200_total_pf, h100_total_pf)
        cpu_pf_ratio = compute_ratio(gh200_row['cpu_page_faults'], h100_row['cpu_page_faults'])
        gpu_pf_ratio = compute_ratio(gh200_row['gpu_page_faults'], h100_row['gpu_page_faults'])
        htod_ratio = compute_ratio(gh200_row['htod_migration_mb'], h100_row['htod_migration_mb'])
        dtoh_ratio = compute_ratio(gh200_row['dtoh_migration_mb'], h100_row['dtoh_migration_mb'])
        
        results.append({
            'workload': workload,
            'pf_count_ratio': pf_count_ratio,
            'cpu_pf_ratio': cpu_pf_ratio,
            'gpu_pf_ratio': gpu_pf_ratio,
            'htod_ratio': htod_ratio,
            'dtoh_ratio': dtoh_ratio,
            'gh200_total_pf': gh200_total_pf,
            'h100_total_pf': h100_total_pf,
            'gh200_cpu_pf': gh200_row['cpu_page_faults'],
            'h100_cpu_pf': h100_row['cpu_page_faults'],
            'gh200_gpu_pf': gh200_row['gpu_page_faults'],
            'h100_gpu_pf': h100_row['gpu_page_faults'],
            'gh200_htod': gh200_row['htod_migration_mb'],
            'h100_htod': h100_row['htod_migration_mb'],
            'gh200_dtoh': gh200_row['dtoh_migration_mb'],
            'h100_dtoh': h100_row['dtoh_migration_mb'],
            'is_dynamic': workload in DYNAMIC_WORKLOADS,
        })
    
    return pd.DataFrame(results)


def plot_comparison_bars(metrics_df, output_path, threads=TARGET_THREADS, gpu_blocks=TARGET_GPU_BLOCKS):
    """Create bar chart comparing H100 vs GH200 page fault count."""
    
    # Sort by workload group order
    metrics_df = metrics_df.copy()
    metrics_df['sort_order'] = metrics_df['workload'].apply(
        lambda w: WORKLOAD_ORDER.index(w) if w in WORKLOAD_ORDER else 999
    )
    metrics_df = metrics_df.sort_values('sort_order')
    
    workloads = metrics_df['workload'].values
    n_workloads = len(workloads)
    
    # Get page fault count ratios
    pf_values = metrics_df['pf_count_ratio'].values.copy()
    
    # Assign a color per bar based on its workload group
    bar_colors = [GROUP_COLORS.get(_workload_to_group(w), '#999999') for w in workloads]
    
    # Bar positioning
    bar_width = 0.6
    x = np.arange(n_workloads)
    
    # Create figure (larger for PPT with bigger fonts)
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot bars (no hatch)
    bars = ax.bar(x, pf_values, bar_width,
                  color=bar_colors,
                  edgecolor='black', linewidth=0.5)
    
    # Add legend entries for each group color
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color, edgecolor='black', label=group)
                      for group, color in GROUP_COLORS.items()]
    legend_handles.append(plt.Line2D([0], [0], color='gray', linestyle='--',
                                     linewidth=2, label='Equal (1x)'))
    
    # Add reference line at 1x (equal)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    
    # Set log scale for y-axis (use symlog to allow 0 origin)
    ax.set_yscale('symlog', linthresh=0.1)
    ax.set_ylim(bottom=0, top=1500)
    
    # Add value labels on top of each bar
    for bar, val in zip(bars, pf_values):
        ax.annotate(f'{val:.1f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Increase bar label font sizes
    ax.tick_params(axis='both', labelsize=20)
    
    # Add vertical lines to separate groups
    group_boundaries = []
    current_pos = 0
    for group_name, group_workloads in WORKLOAD_GROUPS.items():
        # Count how many workloads from this group are in our data
        count = sum(1 for w in group_workloads if w in workloads)
        if count > 0:
            current_pos += count
            group_boundaries.append((current_pos - 0.5, group_name))
    
    # Draw separator lines (except after last group)
    for i, (boundary, _) in enumerate(group_boundaries[:-1]):
        ax.axvline(x=boundary, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Add group labels well below the x-axis tick labels
    current_pos = 0
    for group_name, group_workloads in WORKLOAD_GROUPS.items():
        count = sum(1 for w in group_workloads if w in workloads)
        if count > 0:
            mid_pos = current_pos + count / 2 - 0.5
            ax.text(mid_pos, -0.08, group_name, ha='center', va='top',
                   fontsize=18, fontweight='bold', style='italic',
                   color=GROUP_COLORS.get(group_name, 'black'),
                   transform=ax.get_xaxis_transform())
            current_pos += count
    
    # Configure axes
    ax.set_xlabel('Workload', fontsize=24, fontweight='bold', labelpad=55)
    ax.set_ylabel('Page Fault Count Ratio (H100 SW / GH200 HW)', fontsize=22, fontweight='bold')
    ax.set_title('Page Fault Count: Software Coherence (H100) vs Hardware Coherence (GH200)\n'
                 f'Configuration: threads={threads}, gpu_blocks={gpu_blocks}, '
                 f'partition={TARGET_PARTITION_STATIC} (static) or dynamic',
                 fontsize=24, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha='right', fontsize=20)
    
    ax.tick_params(axis='y', labelsize=20)
    
    # Legend
    ax.legend(handles=legend_handles, loc='upper left', fontsize=18, framealpha=0.9)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close(fig)


def plot_single_metric_bars(metrics_df, output_path, metric_col, metric_label, bar_color, 
                            threads=TARGET_THREADS, gpu_blocks=TARGET_GPU_BLOCKS):
    """Create a single-metric bar chart for CPU or GPU page fault amplification."""
    
    # Cap value for display (lower = shorter bars)
    CAP_VALUE = 8.0
    
    # Sort by workload group order
    metrics_df = metrics_df.copy()
    metrics_df['sort_order'] = metrics_df['workload'].apply(
        lambda w: WORKLOAD_ORDER.index(w) if w in WORKLOAD_ORDER else 999
    )
    metrics_df = metrics_df.sort_values('sort_order')
    
    workloads = metrics_df['workload'].values
    n_workloads = len(workloads)
    
    # Get actual values before capping
    actual_values = metrics_df[metric_col].values.copy()
    
    # Cap values for display
    display_values = np.clip(actual_values, None, CAP_VALUE)
    
    # Bar positioning
    bar_width = 0.6
    x = np.arange(n_workloads)
    
    # Create figure (larger for PPT)
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot bars
    bars = ax.bar(x, display_values, bar_width,
                  label=metric_label,
                  color=bar_color,
                  edgecolor='black', linewidth=0.5)
    
    # Add reference line at 1x (equal)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Equal (1x)')
    
    # Add labels for capped bars showing actual value
    for bar, actual in zip(bars, actual_values):
        if actual > CAP_VALUE:
            ax.annotate(f'{actual:.1f}x',
                       xy=(bar.get_x() + bar.get_width() / 2, CAP_VALUE),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=18, fontweight='bold',
                       rotation=90)
        elif actual >= 1.5:
            # Add value labels for significant bars
            ax.annotate(f'{actual:.1f}x',
                       xy=(bar.get_x() + bar.get_width() / 2, actual),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=16)
    
    # Add vertical lines to separate groups
    group_boundaries = []
    current_pos = 0
    for group_name, group_workloads in WORKLOAD_GROUPS.items():
        count = sum(1 for w in group_workloads if w in workloads)
        if count > 0:
            current_pos += count
            group_boundaries.append((current_pos - 0.5, group_name))
    
    # Draw separator lines (except after last group)
    for i, (boundary, _) in enumerate(group_boundaries[:-1]):
        ax.axvline(x=boundary, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Add group labels at the top
    current_pos = 0
    for group_name, group_workloads in WORKLOAD_GROUPS.items():
        count = sum(1 for w in group_workloads if w in workloads)
        if count > 0:
            mid_pos = current_pos + count / 2 - 0.5
            ax.text(mid_pos, CAP_VALUE + 1.5, group_name, ha='center', va='bottom',
                   fontsize=20, fontweight='bold', style='italic')
            current_pos += count
    
    # Configure axes
    ax.set_xlabel('Workload', fontsize=24, fontweight='bold')
    ax.set_ylabel('Ratio (H100 SW Coherence / GH200 HW Coherence)', fontsize=22, fontweight='bold')
    ax.set_title(f'{metric_label} Amplification: H100 (SW) vs GH200 (HW)\n'
                 f'Configuration: threads={threads}, gpu_blocks={gpu_blocks}',
                 fontsize=24, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha='right', fontsize=20)
    
    # Set y-axis limit
    ax.set_ylim(0, CAP_VALUE + 3.5)
    ax.tick_params(axis='y', labelsize=20)
    
    # Legend
    ax.legend(loc='upper left', fontsize=18, framealpha=0.9)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close(fig)


def print_summary_table(metrics_df, threads=TARGET_THREADS, gpu_blocks=TARGET_GPU_BLOCKS):
    """Print a summary table of the results."""
    print("\n" + "="*120)
    print("PAGE FAULT COMPARISON: H100 (SW Coherence) vs GH200 (HW Coherence)")
    print("="*120)
    print(f"\nConfiguration: threads={threads}, gpu_blocks={gpu_blocks}")
    print(f"Partition: {TARGET_PARTITION_STATIC} for static workloads, 'dynamic' for task-based")
    print("\n" + "-"*120)
    print(f"{'Workload':<10} {'Type':<8} {'Total PF':<12} {'CPU PF':<12} {'GPU PF':<12} {'HtoD Mig':<12} {'DtoH Mig':<12} "
          f"{'GH200 PF':<10} {'H100 PF':<10}")
    print("-"*120)
    
    for _, row in metrics_df.iterrows():
        wtype = "Dynamic" if row['is_dynamic'] else "Static"
        print(f"{row['workload']:<10} {wtype:<8} "
              f"{row['pf_count_ratio']:>10.2f}x "
              f"{row['cpu_pf_ratio']:>10.2f}x "
              f"{row['gpu_pf_ratio']:>10.2f}x "
              f"{row['htod_ratio']:>10.2f}x "
              f"{row['dtoh_ratio']:>10.2f}x "
              f"{row['gh200_total_pf']:>10.0f} "
              f"{row['h100_total_pf']:>10.0f}")
    
    print("-"*120)
    print("\nRatio > 1x = H100 has more page faults/migration than GH200")
    print("Ratio < 1x = H100 has fewer page faults/migration than GH200")
    print("="*120)


def main():
    """Main function to generate comparison plots."""
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data from both systems
    print(f"Loading GH200 data from: {GH200_DIR}")
    gh200_data = load_pagefault_results(GH200_DIR)
    print(f"  Found {len(gh200_data)} workloads")
    
    print(f"Loading H100 data from: {H100_DIR}")
    h100_data = load_pagefault_results(H100_DIR)
    print(f"  Found {len(h100_data)} workloads")
    
    # ========================================
    # Plot 1: Primary config (threads=64, gpu_blocks=64)
    # ========================================
    print(f"\n{'='*60}")
    print(f"Generating plot for threads={TARGET_THREADS}, gpu_blocks={TARGET_GPU_BLOCKS}")
    print(f"{'='*60}")
    
    # Compute comparison metrics
    metrics_df = compute_metrics(gh200_data, h100_data, TARGET_THREADS, TARGET_GPU_BLOCKS)
    
    if metrics_df.empty:
        print("Error: No matching workloads found with target configuration!")
    else:
        # Print summary table
        print_summary_table(metrics_df, TARGET_THREADS, TARGET_GPU_BLOCKS)
        
        # Generate plot
        output_path = OUTPUT_DIR / "pf_comparison_h100_vs_gh200.png"
        plot_comparison_bars(metrics_df, output_path, TARGET_THREADS, TARGET_GPU_BLOCKS)
        
        # Save metrics to CSV for reference
        csv_path = OUTPUT_DIR / "pf_comparison_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"Saved metrics CSV to: {csv_path}")
        
        # ========================================
        # Additional plots for t64_g64: CPU-only and GPU-only PF amplification
        # ========================================
        print(f"\n--- Generating CPU Page Fault amplification plot ---")
        cpu_pf_path = OUTPUT_DIR / "pf_comparison_cpu_only.png"
        plot_single_metric_bars(metrics_df, cpu_pf_path, 
                                'cpu_pf_ratio', 'CPU Page Fault',
                                '#e74c3c',  # Red
                                TARGET_THREADS, TARGET_GPU_BLOCKS)
        
        print(f"\n--- Generating GPU Page Fault amplification plot ---")
        gpu_pf_path = OUTPUT_DIR / "pf_comparison_gpu_only.png"
        plot_single_metric_bars(metrics_df, gpu_pf_path, 
                                'gpu_pf_ratio', 'GPU Page Fault',
                                '#3498db',  # Blue
                                TARGET_THREADS, TARGET_GPU_BLOCKS)
    
    # ========================================
    # Plot 2: Secondary config (threads=4, gpu_blocks=8)
    # ========================================
    print(f"\n{'='*60}")
    print(f"Generating plot for threads={TARGET_THREADS_2}, gpu_blocks={TARGET_GPU_BLOCKS_2}")
    print(f"{'='*60}")
    
    # Compute comparison metrics for second config
    metrics_df_2 = compute_metrics(gh200_data, h100_data, TARGET_THREADS_2, TARGET_GPU_BLOCKS_2)
    
    if metrics_df_2.empty:
        print("Warning: No matching workloads found for secondary configuration!")
    else:
        # Print summary table
        print_summary_table(metrics_df_2, TARGET_THREADS_2, TARGET_GPU_BLOCKS_2)
        
        # Generate plot
        output_path_2 = OUTPUT_DIR / f"pf_comparison_h100_vs_gh200_t{TARGET_THREADS_2}_g{TARGET_GPU_BLOCKS_2}.png"
        plot_comparison_bars(metrics_df_2, output_path_2, TARGET_THREADS_2, TARGET_GPU_BLOCKS_2)
        
        # Save metrics to CSV for reference
        csv_path_2 = OUTPUT_DIR / f"pf_comparison_metrics_t{TARGET_THREADS_2}_g{TARGET_GPU_BLOCKS_2}.csv"
        metrics_df_2.to_csv(csv_path_2, index=False)
        print(f"Saved metrics CSV to: {csv_path_2}")


def generate_pdf_plot(metrics_df, pdf_path, threads, gpu_blocks):
    """Generate a PDF version of the comparison plot."""
    
    # Cap value for display
    CAP_VALUE = 10.0
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Re-create the plot for PDF - sort by group order
    metrics_df_sorted = metrics_df.copy()
    metrics_df_sorted['sort_order'] = metrics_df_sorted['workload'].apply(
        lambda w: WORKLOAD_ORDER.index(w) if w in WORKLOAD_ORDER else 999
    )
    metrics_df_sorted = metrics_df_sorted.sort_values('sort_order')
    workloads = metrics_df_sorted['workload'].values
    n_workloads = len(workloads)
    bar_width = 0.25
    x = np.arange(n_workloads)
    
    # Get actual values before capping
    pf_actual = metrics_df_sorted['pf_count_ratio'].values.copy()
    htod_actual = metrics_df_sorted['htod_ratio'].values.copy()
    dtoh_actual = metrics_df_sorted['dtoh_ratio'].values.copy()
    
    # Cap values for display
    pf_display = np.clip(pf_actual, None, CAP_VALUE)
    htod_display = np.clip(htod_actual, None, CAP_VALUE)
    dtoh_display = np.clip(dtoh_actual, None, CAP_VALUE)
    
    bars1 = ax.bar(x - bar_width, pf_display, bar_width,
                   label='Page Fault Count',
                   color=COLORS['pf_count'], hatch=HATCHES['pf_count'],
                   edgecolor='black', linewidth=0.5)
    
    bars2 = ax.bar(x, htod_display, bar_width,
                   label='HtoD Migration',
                   color=COLORS['htod_mig'], hatch=HATCHES['htod_mig'],
                   edgecolor='black', linewidth=0.5)
    
    bars3 = ax.bar(x + bar_width, dtoh_display, bar_width,
                   label='DtoH Migration',
                   color=COLORS['dtoh_mig'], hatch=HATCHES['dtoh_mig'],
                   edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Equal (1x)')
    
    # Add labels for capped bars
    def add_cap_labels_pdf(bars, actual_values):
        for bar, actual in zip(bars, actual_values):
            if actual > CAP_VALUE:
                ax.annotate(f'{actual:.1f}x',
                           xy=(bar.get_x() + bar.get_width() / 2, CAP_VALUE),
                           xytext=(0, 5),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=11, fontweight='bold',
                           rotation=90)
    
    add_cap_labels_pdf(bars1, pf_actual)
    add_cap_labels_pdf(bars2, htod_actual)
    add_cap_labels_pdf(bars3, dtoh_actual)
    
    # Draw separator lines
    current_pos = 0
    group_boundaries_pdf = []
    for group_name, group_workloads in WORKLOAD_GROUPS.items():
        count = sum(1 for w in group_workloads if w in workloads)
        if count > 0:
            current_pos += count
            group_boundaries_pdf.append((current_pos - 0.5, group_name))
    
    for i, (boundary, _) in enumerate(group_boundaries_pdf[:-1]):
        ax.axvline(x=boundary, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Add group labels
    current_pos = 0
    for group_name, group_workloads in WORKLOAD_GROUPS.items():
        count = sum(1 for w in group_workloads if w in workloads)
        if count > 0:
            mid_pos = current_pos + count / 2 - 0.5
            ax.text(mid_pos, CAP_VALUE + 1.5, group_name, ha='center', va='bottom',
                   fontsize=12, fontweight='bold', style='italic')
            current_pos += count
    
    ax.set_xlabel('Workload', fontsize=14, fontweight='bold')
    ax.set_ylabel('Ratio (H100 SW / GH200 HW Coherence)', fontsize=14, fontweight='bold')
    ax.set_title(f'Page Fault Overhead: Software Coherence (H100) vs Hardware Coherence (GH200)\n'
                 f'Configuration: threads={threads}, gpu_blocks={gpu_blocks}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha='right', fontsize=12)
    ax.set_ylim(0, CAP_VALUE + 3)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF to: {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
