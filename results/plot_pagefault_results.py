#!/usr/bin/env python3
"""
Plot Page Fault Study Results - Similar style to Coherence Study plots
X-axis: CPU/GPU configurations (threads)
Y-axis: Page faults (count) or Page fault data (MB)
Lines: Different partition ratios (-a values) with different markers
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')

# Markers for different partition values (matching paper style)
PARTITION_STYLES = {
    0.0:  {'marker': 's', 'linestyle': '-',  'label': 'GPU only'},
    0.1:  {'marker': 'o', 'linestyle': '-',  'label': 'α=0.1'},
    0.25: {'marker': '^', 'linestyle': '-',  'label': 'α=0.25'},
    0.5:  {'marker': 'D', 'linestyle': '-',  'label': 'α=0.5'},
    0.75: {'marker': 'v', 'linestyle': '-',  'label': 'α=0.75'},
    0.9:  {'marker': '<', 'linestyle': '-',  'label': 'α=0.9'},
    1.0:  {'marker': 'x', 'linestyle': '--', 'label': 'CPU only'},
}

MARKERS = ['s', 'o', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

# Colors for single system plot (viridis palette)
COLORS_SINGLE = plt.cm.viridis(np.linspace(0.1, 0.9, 7))

# Color palettes for comparison (two systems)
# GH200: Blue -> Teal -> Green shades
COLORS_SYSTEM1 = plt.cm.winter(np.linspace(0.1, 0.9, 7))  # Blue-green palette
# x86+H100: Orange -> Pink -> Red shades  
COLORS_SYSTEM2 = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 7))  # Yellow-Orange-Red palette


def load_results(results_dir):
    """Load all CSV results from a directory."""
    data = {}
    csv_files = glob.glob(os.path.join(results_dir, "*_pagefaults.csv"))
    
    for csv_file in csv_files:
        bench_name = os.path.basename(csv_file).replace("_pagefaults.csv", "")
        df = pd.read_csv(csv_file)
        
        # Convert partition to numeric (handle 'dynamic' for task-based benchmarks)
        if 'partition' in df.columns:
            df['partition'] = pd.to_numeric(df['partition'], errors='coerce')
        
        # Convert page fault columns to numeric
        for col in ['cpu_page_faults', 'gpu_page_faults', 'cpu_pf_data_mb', 'gpu_pf_data_mb']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate total page faults
        df['total_page_faults'] = df['cpu_page_faults'].fillna(0) + df['gpu_page_faults'].fillna(0)
        df['total_pf_data_mb'] = df['cpu_pf_data_mb'].fillna(0) + df['gpu_pf_data_mb'].fillna(0)
        
        data[bench_name] = df
    
    return data


def _create_grouped_legend(ax, handles1, labels1, handles2, labels2, name1, name2, loc='best'):
    """Create a grouped legend with system titles."""
    from matplotlib.lines import Line2D
    
    header1 = Line2D([0], [0], color='none', label=f'▬ {name1} (solid)')
    header2 = Line2D([0], [0], color='none', label=f'┄ {name2} (dashed)')
    
    all_handles = [header1] + handles1 + [Line2D([0], [0], color='none', label='')] + [header2] + handles2
    all_labels = [f'▬ {name1}'] + labels1 + [''] + [f'┄ {name2}'] + labels2
    
    ax.legend(all_handles, all_labels, loc=loc, fontsize=10, framealpha=0.9)


def _create_single_legend(ax, handles, labels, name, linestyle_desc, loc='upper right'):
    """Create a legend for a single system."""
    from matplotlib.lines import Line2D
    
    header = Line2D([0], [0], color='none', label=f'{linestyle_desc} {name}')
    all_handles = [header] + handles
    all_labels = [f'{linestyle_desc} {name}'] + labels
    
    ax.legend(all_handles, all_labels, loc=loc, fontsize=10, framealpha=0.9)


def _system_palette(system_name):
    name_lower = system_name.lower()
    if 'h100' in name_lower:
        return COLORS_SYSTEM2
    return COLORS_SYSTEM1


def _ensure_migration_columns(df):
    if 'htod_migration_mb' not in df.columns:
        df['htod_migration_mb'] = df.get('gpu_pf_data_mb', 0)
    if 'dtoh_migration_mb' not in df.columns:
        df['dtoh_migration_mb'] = df.get('cpu_pf_data_mb', 0)


def _normalize_gpu_blocks(df):
    if 'gpu_blocks' not in df.columns:
        return df, False
    numeric_gpu = pd.to_numeric(df['gpu_blocks'], errors='coerce')
    has_gpu_blocks = numeric_gpu.notna().any()
    if not has_gpu_blocks:
        return df, False
    df = df.copy()
    df['gpu_blocks'] = numeric_gpu
    df = df[df['gpu_blocks'].notna()]
    return df, True


def _subplot_grid(n_subplots):
    if n_subplots <= 2:
        return 1, n_subplots
    if n_subplots <= 4:
        return 2, 2
    return 2, 3


def _pick_palette_colors(palette, count):
    if count <= 0:
        return np.array([])
    if count == 1:
        return np.array([palette[len(palette)//2]])
    idx = np.linspace(0, len(palette) - 1, count).astype(int)
    return palette[idx]


def _add_partition_and_metric_legends(ax, metric_styles, partitions, partition_colors):
    from matplotlib.lines import Line2D

    shape_handles = [
        Line2D([0], [0], marker=style['marker'], color='gray',
               linestyle='None', markersize=8, label=style['label'])
        for style in metric_styles.values()
    ]

    partition_handles = [
        Line2D([0], [0], marker='o', color=partition_colors[i],
               linestyle='-', linewidth=2, markersize=8,
               label=PARTITION_STYLES.get(p, {'label': f'α={p}'})['label'])
        for i, p in enumerate(partitions)
    ]

    legend1 = ax.legend(handles=shape_handles, loc='upper left', fontsize=9,
                        title='Metric Type', framealpha=0.9)
    ax.add_artist(legend1)
    ax.legend(handles=partition_handles, loc='upper right', fontsize=9,
              title='Partition (α)', framealpha=0.9)


def plot_benchmark_dual_axis(df, bench_name, output_dir, system_name="System"):
    """
    Plot page faults and migration data on separate plots.
    Plot 1: CPU vs GPU page faults
    Plot 2: HtoD vs DtoH migration volume
    """
    df = df.copy()
    _ensure_migration_columns(df)
    df, has_gpu_blocks = _normalize_gpu_blocks(df)
    has_partitions = 'partition' in df.columns and df['partition'].notna().any()

    threads = sorted(df['threads'].unique())
    palette = _system_palette(system_name)

    pf_styles = {
        'cpu_page_faults': {'marker': 'o', 'label': 'CPU Page Faults'},
        'gpu_page_faults': {'marker': 's', 'label': 'GPU Page Faults'},
    }

    mig_styles = {
        'htod_migration_mb': {'marker': 'o', 'label': 'HtoD (GPU←CPU)'},
        'dtoh_migration_mb': {'marker': 's', 'label': 'DtoH (CPU←GPU)'},
    }

    def plot_metric_set(metric_styles, ylabel, title_suffix, filename_suffix):
        if has_gpu_blocks:
            gpu_blocks_vals = sorted(df['gpu_blocks'].dropna().unique())
            nrows, ncols = _subplot_grid(len(gpu_blocks_vals))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), squeeze=False)
            axes = axes.flatten()
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            axes = [ax]
            gpu_blocks_vals = [None]

        for idx, g in enumerate(gpu_blocks_vals):
            ax = axes[idx]
            subset_g = df if g is None else df[df['gpu_blocks'] == g]

            if has_partitions:
                partitions = sorted(subset_g['partition'].dropna().unique())
                if partitions:
                    partition_colors = _pick_palette_colors(palette, len(partitions))
                    for p_idx, p in enumerate(partitions):
                        subset = subset_g[subset_g['partition'] == p].sort_values('threads')
                        if len(subset) == 0:
                            continue
                        for metric, style in metric_styles.items():
                            ax.plot(subset['threads'], subset[metric],
                                    marker=style['marker'],
                                    linestyle='-',
                                    color=partition_colors[p_idx],
                                    linewidth=2, markersize=8, alpha=0.9)

                    if idx == 0:
                        _add_partition_and_metric_legends(ax, metric_styles, partitions, partition_colors)
                else:
                    subset = subset_g.sort_values('threads')
                    for metric, style in metric_styles.items():
                        ax.plot(subset['threads'], subset[metric],
                                marker=style['marker'],
                                linestyle='-',
                                color=palette[3],
                                linewidth=2, markersize=8,
                                label=style['label'])
                    if idx == 0:
                        ax.legend(loc='best', fontsize=10)
            else:
                subset = subset_g.sort_values('threads')
                metric_colors = _pick_palette_colors(palette, len(metric_styles))
                for m_idx, (metric, style) in enumerate(metric_styles.items()):
                    ax.plot(subset['threads'], subset[metric],
                            marker=style['marker'],
                            linestyle='-',
                            color=metric_colors[m_idx],
                            linewidth=2, markersize=8,
                            label=style['label'])
                if idx == 0:
                    ax.legend(loc='best', fontsize=10)

            ax.set_xlabel('CPU Threads', fontsize=10 if has_gpu_blocks else 12)
            ax.set_ylabel(ylabel, fontsize=10 if has_gpu_blocks else 12)
            if g is not None:
                ax.set_title(f'GPU Blocks = {int(g)}', fontsize=11, fontweight='bold')
            ax.set_xticks(threads)
            ax.grid(True, alpha=0.3)

        for idx in range(len(gpu_blocks_vals), len(axes)):
            axes[idx].axis('off')

        if has_gpu_blocks:
            plt.suptitle(f'{bench_name} - {title_suffix}', fontsize=14, fontweight='bold')
        else:
            axes[0].set_title(f'{bench_name} - {title_suffix}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{bench_name}_{filename_suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    plot_metric_set(pf_styles, 'Page Faults (count)', 'Page Faults', 'pagefaults')
    plot_metric_set(mig_styles, 'Migration Volume (MB)', 'Data Migration', 'migration')

    return True


def plot_comparison_dual_axis(df1, df2, bench_name, output_dir, name1="GH200", name2="x86+H100"):
    """
    Compare two systems with separate page fault and migration plots.
    """
    from matplotlib.lines import Line2D

    df1 = df1.copy()
    df2 = df2.copy()
    _ensure_migration_columns(df1)
    _ensure_migration_columns(df2)

    df1['total_page_faults'] = df1['cpu_page_faults'].fillna(0) + df1['gpu_page_faults'].fillna(0)
    df2['total_page_faults'] = df2['cpu_page_faults'].fillna(0) + df2['gpu_page_faults'].fillna(0)
    df1['total_migration_mb'] = df1['htod_migration_mb'].fillna(0) + df1['dtoh_migration_mb'].fillna(0)
    df2['total_migration_mb'] = df2['htod_migration_mb'].fillna(0) + df2['dtoh_migration_mb'].fillna(0)

    df1, has_gpu_blocks1 = _normalize_gpu_blocks(df1)
    df2, has_gpu_blocks2 = _normalize_gpu_blocks(df2)
    has_gpu_blocks = has_gpu_blocks1 and has_gpu_blocks2

    has_partitions = ('partition' in df1.columns and df1['partition'].notna().any()) or \
                     ('partition' in df2.columns and df2['partition'].notna().any())
    threads = sorted(set(df1['threads'].unique()) | set(df2['threads'].unique()))

    def plot_sum_comparison(metric, ylabel, title_suffix, filename_suffix):
        if has_gpu_blocks:
            gpu_blocks_vals = sorted(set(df1['gpu_blocks'].dropna().unique()) |
                                     set(df2['gpu_blocks'].dropna().unique()))
            nrows, ncols = _subplot_grid(len(gpu_blocks_vals))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), squeeze=False)
            axes = axes.flatten()
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            axes = [ax]
            gpu_blocks_vals = [None]

        for idx, g in enumerate(gpu_blocks_vals):
            ax = axes[idx]
            subset1 = df1 if g is None else df1[df1['gpu_blocks'] == g]
            subset2 = df2 if g is None else df2[df2['gpu_blocks'] == g]

            if has_partitions:
                partitions = sorted(set(subset1['partition'].dropna().unique()) |
                                   set(subset2['partition'].dropna().unique()))
                if partitions:
                    colors1 = _pick_palette_colors(COLORS_SYSTEM1, len(partitions))
                    colors2 = _pick_palette_colors(COLORS_SYSTEM2, len(partitions))

                    for p_idx, p in enumerate(partitions):
                        s1 = subset1[subset1['partition'] == p].sort_values('threads')
                        if len(s1) > 0:
                            ax.plot(s1['threads'], s1[metric],
                                    marker='o', linestyle='-',
                                    color=colors1[p_idx], linewidth=2.5, markersize=8)
                        s2 = subset2[subset2['partition'] == p].sort_values('threads')
                        if len(s2) > 0:
                            ax.plot(s2['threads'], s2[metric],
                                    marker='o', linestyle=':',
                                    color=colors2[p_idx], linewidth=2.5, markersize=8)

                    if idx == 0:
                        partition_handles = [
                            Line2D([0], [0], marker='o', color=colors1[i],
                                   linestyle='-', linewidth=2, markersize=8,
                                   label=PARTITION_STYLES.get(p, {'label': f'α={p}'})['label'])
                            for i, p in enumerate(partitions)
                        ]
                        style_handles = [
                            Line2D([0], [0], color='black', linestyle='-', label=name1),
                            Line2D([0], [0], color='black', linestyle=':', label=name2),
                        ]
                        legend1 = ax.legend(handles=style_handles, loc='upper left', fontsize=9,
                                            title='System', framealpha=0.9)
                        ax.add_artist(legend1)
                        ax.legend(handles=partition_handles, loc='upper right', fontsize=9,
                                  title='Partition (α)', framealpha=0.9)
                else:
                    s1 = subset1.sort_values('threads')
                    s2 = subset2.sort_values('threads')
                    ax.plot(s1['threads'], s1[metric], marker='o', linestyle='-',
                            color=COLORS_SYSTEM1[3], linewidth=2.5, markersize=8, label=name1)
                    ax.plot(s2['threads'], s2[metric], marker='o', linestyle=':',
                            color=COLORS_SYSTEM2[3], linewidth=2.5, markersize=8, label=name2)
                    if idx == 0:
                        ax.legend(loc='best', fontsize=10)
            else:
                s1 = subset1.sort_values('threads')
                s2 = subset2.sort_values('threads')
                ax.plot(s1['threads'], s1[metric], marker='o', linestyle='-',
                        color=COLORS_SYSTEM1[3], linewidth=2.5, markersize=8, label=name1)
                ax.plot(s2['threads'], s2[metric], marker='o', linestyle=':',
                        color=COLORS_SYSTEM2[3], linewidth=2.5, markersize=8, label=name2)
                if idx == 0:
                    ax.legend(loc='best', fontsize=10)

            ax.set_xlabel('CPU Threads', fontsize=10 if has_gpu_blocks else 12)
            ax.set_ylabel(ylabel, fontsize=10 if has_gpu_blocks else 12)
            if g is not None:
                ax.set_title(f'GPU Blocks = {int(g)}', fontsize=11, fontweight='bold')
            ax.set_xticks(threads)
            ax.grid(True, alpha=0.3)

        for idx in range(len(gpu_blocks_vals), len(axes)):
            axes[idx].axis('off')

        if has_gpu_blocks:
            plt.suptitle(f'{bench_name} - {title_suffix}', fontsize=14, fontweight='bold')
        else:
            axes[0].set_title(f'{bench_name} - {title_suffix}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{bench_name}_{filename_suffix}_comparison.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

    plot_sum_comparison('total_page_faults', 'Page Faults (count)', 'Page Faults Comparison', 'pagefaults')
    plot_sum_comparison('total_migration_mb', 'Migration Volume (MB)', 'Data Migration Comparison', 'migration')


def plot_benchmark_pagefaults(df, bench_name, output_dir, system_name="System", metric='total_page_faults'):
    """
    Plot page faults for a single benchmark.
    metric: 'cpu_page_faults', 'gpu_page_faults', 'total_page_faults', 
            'cpu_pf_data_mb', 'gpu_pf_data_mb', 'total_pf_data_mb'
    """
    
    metric_labels = {
        'cpu_page_faults': 'CPU Page Faults',
        'gpu_page_faults': 'GPU Page Faults', 
        'total_page_faults': 'Total Page Faults',
        'cpu_pf_data_mb': 'CPU Page Fault Data (MB)',
        'gpu_pf_data_mb': 'GPU Page Fault Data (MB)',
        'total_pf_data_mb': 'Total Page Fault Data (MB)',
    }
    
    y_label = metric_labels.get(metric, metric)
    
    # Check if we have numeric gpu_blocks column
    has_gpu_blocks = False
    if 'gpu_blocks' in df.columns:
        numeric_gpu = pd.to_numeric(df['gpu_blocks'], errors='coerce')
        has_gpu_blocks = numeric_gpu.notna().any()
        if has_gpu_blocks:
            df = df.copy()
            df['gpu_blocks'] = numeric_gpu
    
    # Check if partition-based
    has_partitions = 'partition' in df.columns and df['partition'].notna().any()
    
    if has_gpu_blocks and has_partitions:
        # Subplots for each GPU block count
        gpu_blocks_vals = sorted(df['gpu_blocks'].dropna().unique())
        n_subplots = len(gpu_blocks_vals)
        
        if n_subplots <= 2:
            nrows, ncols = 1, n_subplots
        elif n_subplots <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), squeeze=False)
        axes = axes.flatten()
        
        partitions = sorted(df['partition'].dropna().unique())
        threads = sorted(df['threads'].unique())
        
        for idx, g in enumerate(gpu_blocks_vals):
            ax = axes[idx]
            subset_g = df[df['gpu_blocks'] == g]
            
            for i, p in enumerate(partitions):
                subset = subset_g[subset_g['partition'] == p].sort_values('threads')
                
                if len(subset) == 0:
                    continue
                
                style = PARTITION_STYLES.get(p, {'marker': 'o', 'linestyle': '-', 'label': f'α={p}'})
                
                ax.plot(subset['threads'], subset[metric],
                       marker=style['marker'],
                       linestyle=style['linestyle'],
                       color=COLORS_SINGLE[i % len(COLORS_SINGLE)],
                       linewidth=2, markersize=8,
                       label=style['label'])
            
            ax.set_xlabel('CPU Threads', fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(f'GPU Blocks = {int(g)}', fontsize=11, fontweight='bold')
            ax.set_xticks(threads)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(title='Partition (α)', loc='best', fontsize=8)
        
        for idx in range(len(gpu_blocks_vals), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{bench_name} - {system_name}', fontsize=14, fontweight='bold')
        
    elif has_partitions:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        partitions = sorted(df['partition'].dropna().unique())
        threads = sorted(df['threads'].unique())
        
        for i, p in enumerate(partitions):
            subset = df[df['partition'] == p].sort_values('threads')
            
            if len(subset) == 0:
                continue
            
            style = PARTITION_STYLES.get(p, {'marker': 'o', 'linestyle': '-', 'label': f'α={p}'})
            
            ax.plot(subset['threads'], subset[metric],
                   marker=style['marker'],
                   linestyle=style['linestyle'],
                   color=COLORS_SINGLE[i % len(COLORS_SINGLE)],
                   linewidth=2, markersize=8,
                   label=style['label'])
        
        ax.set_xlabel('CPU Threads', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{bench_name} - {system_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(threads)
        ax.legend(title='Partition (α)', loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if has_gpu_blocks:
            gpu_blocks_vals = sorted(df['gpu_blocks'].dropna().unique())
            threads = sorted(df['threads'].unique())
            
            for i, g in enumerate(gpu_blocks_vals):
                subset = df[df['gpu_blocks'] == g].sort_values('threads')
                ax.plot(subset['threads'], subset[metric],
                       marker=MARKERS[i % len(MARKERS)],
                       color=COLORS_SINGLE[i % len(COLORS_SINGLE)],
                       linewidth=2, markersize=8,
                       label=f'GPU({int(g)} blocks)')
            
            ax.set_xticks(threads)
            ax.legend(title='GPU Blocks', loc='best', fontsize=9)
        else:
            threads = sorted(df['threads'].unique())
            subset = df.sort_values('threads')
            ax.plot(subset['threads'], subset[metric],
                   marker='o', color=COLORS_SINGLE[0],
                   linewidth=2, markersize=8,
                   label='Dynamic')
            ax.set_xticks(threads)
            ax.legend(loc='best')
        
        ax.set_xlabel('CPU Threads', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{bench_name} - {system_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with metric in filename
    metric_short = metric.replace('_', '')
    plt.savefig(os.path.join(output_dir, f'{bench_name}_{metric_short}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def plot_comparison_pagefaults(df1, df2, bench_name, output_dir, name1="GH200", name2="x86+H100", 
                                metric='total_page_faults'):
    """
    Compare page faults between two systems.
    """
    
    metric_labels = {
        'cpu_page_faults': 'CPU Page Faults',
        'gpu_page_faults': 'GPU Page Faults', 
        'total_page_faults': 'Total Page Faults',
        'cpu_pf_data_mb': 'CPU Page Fault Data (MB)',
        'gpu_pf_data_mb': 'GPU Page Fault Data (MB)',
        'total_pf_data_mb': 'Total Page Fault Data (MB)',
    }
    
    y_label = metric_labels.get(metric, metric)
    
    # Check for GPU blocks
    has_gpu_blocks = False
    if 'gpu_blocks' in df1.columns:
        numeric_gpu1 = pd.to_numeric(df1['gpu_blocks'], errors='coerce')
        numeric_gpu2 = pd.to_numeric(df2['gpu_blocks'], errors='coerce')
        has_gpu_blocks = numeric_gpu1.notna().any() and numeric_gpu2.notna().any()
        if has_gpu_blocks:
            df1 = df1.copy()
            df2 = df2.copy()
            df1['gpu_blocks'] = numeric_gpu1
            df2['gpu_blocks'] = numeric_gpu2
            df1 = df1[df1['gpu_blocks'].notna()]
            df2 = df2[df2['gpu_blocks'].notna()]
    
    has_partitions = 'partition' in df1.columns and df1['partition'].notna().any()
    
    if has_gpu_blocks and has_partitions:
        gpu_blocks_vals = sorted(set(df1['gpu_blocks'].dropna().unique()) | 
                                  set(df2['gpu_blocks'].dropna().unique()))
        n_subplots = len(gpu_blocks_vals)
        
        if n_subplots <= 2:
            nrows, ncols = 1, n_subplots
        elif n_subplots <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), squeeze=False)
        axes = axes.flatten()
        
        partitions = sorted(set(df1['partition'].dropna().unique()) | 
                           set(df2['partition'].dropna().unique()))
        threads = sorted(set(df1['threads'].unique()) | set(df2['threads'].unique()))
        
        for idx, g in enumerate(gpu_blocks_vals):
            ax = axes[idx]
            subset_g1 = df1[df1['gpu_blocks'] == g]
            subset_g2 = df2[df2['gpu_blocks'] == g]
            
            handles1, labels1 = [], []
            handles2, labels2 = [], []
            
            for i, p in enumerate(partitions):
                style = PARTITION_STYLES.get(p, {'marker': 'o', 'linestyle': '-', 'label': f'α={p}'})
                
                subset1 = subset_g1[subset_g1['partition'] == p].sort_values('threads')
                if len(subset1) > 0:
                    line1, = ax.plot(subset1['threads'], subset1[metric],
                           marker=style['marker'],
                           linestyle='-',
                           color=COLORS_SYSTEM1[i % len(COLORS_SYSTEM1)],
                           linewidth=2, markersize=8)
                    handles1.append(line1)
                    labels1.append(style['label'])
                
                subset2 = subset_g2[subset_g2['partition'] == p].sort_values('threads')
                if len(subset2) > 0:
                    line2, = ax.plot(subset2['threads'], subset2[metric],
                           marker=style['marker'],
                           linestyle='--',
                           color=COLORS_SYSTEM2[i % len(COLORS_SYSTEM2)],
                           linewidth=2, markersize=8)
                    handles2.append(line2)
                    labels2.append(style['label'])
            
            ax.set_xlabel('CPU Threads', fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(f'GPU Blocks = {int(g)}', fontsize=11, fontweight='bold')
            ax.set_xticks(threads)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                first_handles1, first_labels1 = handles1.copy(), labels1.copy()
                first_handles2, first_labels2 = handles2.copy(), labels2.copy()
        
        _create_single_legend(axes[0], first_handles1, first_labels1, name1, '▬', loc='upper right')
        if len(gpu_blocks_vals) > 1:
            _create_single_legend(axes[1], first_handles2, first_labels2, name2, '┄', loc='upper right')
        
        for idx in range(len(gpu_blocks_vals), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{bench_name} - Page Faults', fontsize=14, fontweight='bold')
        
    elif has_partitions:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        df1 = df1.copy()
        df2 = df2.copy()
        
        partitions = sorted(set(df1['partition'].dropna().unique()) | 
                           set(df2['partition'].dropna().unique()))
        threads = sorted(set(df1['threads'].unique()) | set(df2['threads'].unique()))
        
        handles1, labels1 = [], []
        handles2, labels2 = [], []
        
        for i, p in enumerate(partitions):
            style = PARTITION_STYLES.get(p, {'marker': 'o', 'linestyle': '-', 'label': f'α={p}'})
            
            subset1 = df1[df1['partition'] == p].sort_values('threads')
            if len(subset1) > 0:
                line1, = ax.plot(subset1['threads'], subset1[metric],
                       marker=style['marker'],
                       linestyle='-',
                       color=COLORS_SYSTEM1[i % len(COLORS_SYSTEM1)],
                       linewidth=2, markersize=8)
                handles1.append(line1)
                labels1.append(style['label'])
            
            subset2 = df2[df2['partition'] == p].sort_values('threads')
            if len(subset2) > 0:
                line2, = ax.plot(subset2['threads'], subset2[metric],
                       marker=style['marker'],
                       linestyle='--',
                       color=COLORS_SYSTEM2[i % len(COLORS_SYSTEM2)],
                       linewidth=2, markersize=8)
                handles2.append(line2)
                labels2.append(style['label'])
        
        ax.set_xlabel('CPU Threads', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{bench_name} - Page Faults', fontsize=14, fontweight='bold')
        ax.set_xticks(threads)
        _create_grouped_legend(ax, handles1, labels1, handles2, labels2, name1, name2, loc='upper left')
        ax.grid(True, alpha=0.3)
        
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        df1 = df1.copy()
        df2 = df2.copy()
        
        if has_gpu_blocks:
            gpu_blocks_vals = sorted(set(df1['gpu_blocks'].dropna().unique()) | 
                                      set(df2['gpu_blocks'].dropna().unique()))
            threads = sorted(set(df1['threads'].unique()) | set(df2['threads'].unique()))
            
            handles1, labels1 = [], []
            handles2, labels2 = [], []
            
            for i, g in enumerate(gpu_blocks_vals):
                subset1 = df1[df1['gpu_blocks'] == g].sort_values('threads')
                if len(subset1) > 0:
                    line1, = ax.plot(subset1['threads'], subset1[metric],
                           marker=MARKERS[i % len(MARKERS)],
                           linestyle='-',
                           color=COLORS_SYSTEM1[i % len(COLORS_SYSTEM1)],
                           linewidth=2.5, markersize=10)
                    handles1.append(line1)
                    labels1.append(f'GPU ({int(g)} blocks)')
                
                subset2 = df2[df2['gpu_blocks'] == g].sort_values('threads')
                if len(subset2) > 0:
                    line2, = ax.plot(subset2['threads'], subset2[metric],
                           marker=MARKERS[i % len(MARKERS)],
                           linestyle='--',
                           color=COLORS_SYSTEM2[i % len(COLORS_SYSTEM2)],
                           linewidth=2.5, markersize=10)
                    handles2.append(line2)
                    labels2.append(f'GPU ({int(g)} blocks)')
            
            ax.set_xticks(threads)
            _create_grouped_legend(ax, handles1, labels1, handles2, labels2, name1, name2, loc='upper left')
        else:
            threads = sorted(set(df1['threads'].unique()) | set(df2['threads'].unique()))
            
            subset1 = df1.sort_values('threads')
            subset2 = df2.sort_values('threads')
            
            line1, = ax.plot(subset1['threads'], subset1[metric],
                   marker='o', linestyle='-',
                   color=COLORS_SYSTEM1[3],
                   linewidth=2.5, markersize=10)
            line2, = ax.plot(subset2['threads'], subset2[metric],
                   marker='o', linestyle='--',
                   color=COLORS_SYSTEM2[3],
                   linewidth=2.5, markersize=10)
            
            ax.set_xticks(threads)
            _create_grouped_legend(ax, [line1], ['Dynamic'], [line2], ['Dynamic'], name1, name2, loc='upper left')
        
        ax.set_xlabel('CPU Threads', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_title(f'{bench_name} - Page Faults', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    metric_short = metric.replace('_', '')
    plt.savefig(os.path.join(output_dir, f'{bench_name}_{metric_short}_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_pagefault_reduction(df1, df2, bench_name, output_dir, name1="GH200", name2="x86+H100"):
    """
    Plot page fault reduction ratio (system2 / system1).
    Values > 1 mean system1 has fewer page faults (wins).
    """
    
    # Merge dataframes on common keys
    merge_keys = ['threads']
    if 'gpu_blocks' in df1.columns and 'gpu_blocks' in df2.columns:
        merge_keys.append('gpu_blocks')
    if 'partition' in df1.columns and 'partition' in df2.columns:
        merge_keys.append('partition')
    
    df1_clean = df1.copy()
    df2_clean = df2.copy()
    
    # Convert to numeric for merging
    for col in merge_keys:
        if col in df1_clean.columns:
            df1_clean[col] = pd.to_numeric(df1_clean[col], errors='coerce')
        if col in df2_clean.columns:
            df2_clean[col] = pd.to_numeric(df2_clean[col], errors='coerce')
    
    merged = pd.merge(df1_clean, df2_clean, on=merge_keys, suffixes=('_1', '_2'), how='inner')
    
    if len(merged) == 0:
        print(f"  Warning: No matching data points for {bench_name}")
        return
    
    # Calculate reduction ratio (system2 / system1)
    merged['pf_reduction'] = merged['total_page_faults_2'] / merged['total_page_faults_1'].replace(0, np.nan)
    merged = merged.dropna(subset=['pf_reduction'])
    
    if len(merged) == 0:
        print(f"  Warning: No valid reduction ratios for {bench_name}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    has_partitions = 'partition' in merged.columns and merged['partition'].notna().any()
    
    if has_partitions:
        partitions = sorted(merged['partition'].dropna().unique())
        threads = sorted(merged['threads'].unique())
        
        for i, p in enumerate(partitions):
            subset = merged[merged['partition'] == p].sort_values('threads')
            
            if len(subset) == 0:
                continue
            
            style = PARTITION_STYLES.get(p, {'marker': 'o', 'linestyle': '-', 'label': f'α={p}'})
            
            ax.plot(subset['threads'], subset['pf_reduction'],
                   marker=style['marker'],
                   linestyle=style['linestyle'],
                   color=COLORS_SINGLE[i % len(COLORS_SINGLE)],
                   linewidth=2, markersize=8,
                   label=style['label'])
        
        ax.set_xticks(threads)
        ax.legend(title='Partition (α)', loc='best', fontsize=9)
    else:
        threads = sorted(merged['threads'].unique())
        subset = merged.sort_values('threads')
        ax.plot(subset['threads'], subset['pf_reduction'],
               marker='o', color=COLORS_SINGLE[0],
               linewidth=2, markersize=8)
        ax.set_xticks(threads)
    
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Break-even')
    ax.set_xlabel('CPU Threads', fontsize=12)
    ax.set_ylabel(f'Page Fault Reduction ({name2}/{name1})', fontsize=12)
    ax.set_title(f'{bench_name} - Page Fault Reduction\n(>1 means {name1} wins)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{bench_name}_pf_reduction.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_benchmark_dual_comparison(df1, df2, bench_name, output_dir, name1="GH200", name2="x86+H100"):
    """
    Compare migration volume between two systems.
    """
    from matplotlib.lines import Line2D
    
    # Ensure we have migration columns
    for df in [df1, df2]:
        if 'htod_migration_mb' not in df.columns:
            df['htod_migration_mb'] = 0
        if 'dtoh_migration_mb' not in df.columns:
            df['dtoh_migration_mb'] = 0
        df['total_migration_mb'] = df['htod_migration_mb'].fillna(0) + df['dtoh_migration_mb'].fillna(0)
    
    threads = sorted(set(df1['threads'].unique()) | set(df2['threads'].unique()))
    
    # Define migration metrics
    mig_styles = {
        'htod_migration_mb': {'marker': 'o', 'color': 'purple', 'label': 'HtoD (GPU←CPU)'},
        'dtoh_migration_mb': {'marker': 's', 'color': 'orange', 'label': 'DtoH (CPU←GPU)'},
        'total_migration_mb': {'marker': '^', 'color': 'brown', 'label': 'Total Migration'},
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    subset1 = df1.sort_values('threads')
    subset2 = df2.sort_values('threads')
    
    colors1 = {'htod_migration_mb': '#9467bd', 'dtoh_migration_mb': '#ff7f0e', 'total_migration_mb': '#8c564b'}
    colors2 = {'htod_migration_mb': '#c5b0d5', 'dtoh_migration_mb': '#ffbb78', 'total_migration_mb': '#c49c94'}
    
    for metric, style in mig_styles.items():
        ax.plot(subset1['threads'], subset1[metric],
               marker=style['marker'],
               linestyle='-',
               color=colors1[metric],
               linewidth=2, markersize=8,
               label=f"{style['label']} ({name1})")
        
        ax.plot(subset2['threads'], subset2[metric],
               marker=style['marker'],
               linestyle='--',
               color=colors2[metric],
               linewidth=2, markersize=8,
               label=f"{style['label']} ({name2})")
    
    ax.set_xlabel('CPU Threads', fontsize=12)
    ax.set_ylabel('Migration Volume (MB)', fontsize=12)
    ax.set_title(f'{bench_name} - Data Migration Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(threads)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{bench_name}_migration_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_all_benchmarks(data, output_dir, system_name="System"):
    """Generate page fault and migration plots for all benchmarks."""
    os.makedirs(output_dir, exist_ok=True)
    
    for bench_name, df in data.items():
        print(f"  Plotting {bench_name}...")
        
        # Plot total page faults
        plot_benchmark_pagefaults(df, bench_name, output_dir, system_name, 'total_page_faults')
        
        # Plot CPU page faults
        plot_benchmark_pagefaults(df, bench_name, output_dir, system_name, 'cpu_page_faults')
        
        # Plot GPU page faults
        plot_benchmark_pagefaults(df, bench_name, output_dir, system_name, 'gpu_page_faults')
        
        # Plot migration volumes (if data available)
        if 'htod_migration_mb' in df.columns or 'dtoh_migration_mb' in df.columns:
            plot_benchmark_dual_axis(df, bench_name, output_dir, system_name)
        
        print(f"    Saved {bench_name} plots")


def plot_all_comparisons(data1, data2, output_dir, name1="GH200", name2="x86+H100"):
    """Generate comparison plots for page faults and migration for all common benchmarks."""
    os.makedirs(output_dir, exist_ok=True)
    
    common_benchmarks = set(data1.keys()) & set(data2.keys())
    
    if not common_benchmarks:
        print("Error: No common benchmarks found between the two directories")
        return
    
    for bench_name in sorted(common_benchmarks):
        print(f"  Plotting {bench_name} comparison...")
        
        df1 = data1[bench_name]
        df2 = data2[bench_name]

        plot_comparison_dual_axis(df1, df2, bench_name, output_dir, name1, name2)
        plot_pagefault_reduction(df1, df2, bench_name, output_dir, name1, name2)
        
        print(f"    Saved {bench_name} comparison plots")


def plot_summary_bar_chart(data1, data2, output_dir, name1="GH200", name2="x86+H100"):
    """
    Create summary bar charts comparing total page faults and migration volume across all benchmarks.
    Useful for paper figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    common_benchmarks = sorted(set(data1.keys()) & set(data2.keys()))
    
    if not common_benchmarks:
        return
    
    # =====================
    # Plot 1: Page Faults Summary
    # =====================
    
    # Aggregate total page faults per benchmark (sum across all configs)
    totals1_pf = []
    totals2_pf = []
    
    for bench in common_benchmarks:
        total1 = data1[bench]['total_page_faults'].sum()
        total2 = data2[bench]['total_page_faults'].sum()
        totals1_pf.append(total1)
        totals2_pf.append(total2)
    
    x = np.arange(len(common_benchmarks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width/2, totals1_pf, width, label=name1, color=COLORS_SYSTEM1[3])
    bars2 = ax.bar(x + width/2, totals2_pf, width, label=name2, color=COLORS_SYSTEM2[3])
    
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Total Page Faults (sum across configs)', fontsize=12)
    ax.set_title('Page Fault Comparison Across Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_benchmarks, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add reduction percentage labels
    for i, (t1, t2) in enumerate(zip(totals1_pf, totals2_pf)):
        if t1 > 0:
            reduction = (t2 - t1) / t2 * 100 if t2 > 0 else 0
            if reduction > 0:
                ax.annotate(f'{reduction:.0f}%↓', 
                           xy=(x[i], max(t1, t2) * 1.05),
                           ha='center', fontsize=8, color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pagefault_summary.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved page fault summary bar chart")
    
    # =====================
    # Plot 2: Migration Volume Summary
    # =====================
    
    # Aggregate total migration per benchmark
    totals1_mig = []
    totals2_mig = []
    
    for bench in common_benchmarks:
        # Sum HtoD and DtoH migration volumes
        df1 = data1[bench]
        df2 = data2[bench]
        
        # Handle missing columns
        htod1 = df1.get('htod_migration_mb', pd.Series(0)).fillna(0).sum()
        dtoh1 = df1.get('dtoh_migration_mb', pd.Series(0)).fillna(0).sum()
        total1 = htod1 + dtoh1
        
        htod2 = df2.get('htod_migration_mb', pd.Series(0)).fillna(0).sum()
        dtoh2 = df2.get('dtoh_migration_mb', pd.Series(0)).fillna(0).sum()
        total2 = htod2 + dtoh2
        
        totals1_mig.append(total1)
        totals2_mig.append(total2)
    
    # Only plot if we have meaningful migration data
    if max(max(totals1_mig), max(totals2_mig)) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width/2, totals1_mig, width, label=name1, color=COLORS_SYSTEM1[2])
        bars2 = ax.bar(x + width/2, totals2_mig, width, label=name2, color=COLORS_SYSTEM2[2])
        
        ax.set_xlabel('Benchmark', fontsize=12)
        ax.set_ylabel('Total Migration Volume (MB)', fontsize=12)
        ax.set_title('Data Migration Volume Comparison Across Benchmarks', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(common_benchmarks, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add reduction percentage labels
        for i, (t1, t2) in enumerate(zip(totals1_mig, totals2_mig)):
            if t2 > 0:
                reduction = (t2 - t1) / t2 * 100
                if reduction > 0:
                    ax.annotate(f'{reduction:.0f}%↓', 
                               xy=(x[i], max(t1, t2) * 1.05),
                               ha='center', fontsize=8, color='green')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'migration_summary.png'), dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved migration volume summary bar chart")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single system:  python3 plot_pagefault_results.py <results_dir> [system_name]")
        print("  Comparison:     python3 plot_pagefault_results.py <dir1> <dir2> <name1> <name2>")
        print("")
        print("Examples:")
        print("  python3 plot_pagefault_results.py gh200_results 'GH200'")
        print("  python3 plot_pagefault_results.py gh200_results x86_results 'GH200' 'x86+H100'")
        sys.exit(1)
    
    if len(sys.argv) == 2 or len(sys.argv) == 3:
        # Single system mode
        results_dir = sys.argv[1]
        system_name = sys.argv[2] if len(sys.argv) == 3 else "System"
        
        print(f"Loading results from {results_dir}...")
        data = load_results(results_dir)
        
        if not data:
            print(f"No *_pagefaults.csv files found in {results_dir}")
            sys.exit(1)
        
        print(f"Found {len(data)} benchmarks: {', '.join(data.keys())}")
        
        output_dir = os.path.join(results_dir, 'plots')
        print(f"Generating plots in {output_dir}...")
        plot_all_benchmarks(data, output_dir, system_name)
        
        print(f"\nDone! Plots saved to {output_dir}")
        
    else:
        # Comparison mode
        dir1 = sys.argv[1]
        dir2 = sys.argv[2]
        name1 = sys.argv[3] if len(sys.argv) > 3 else "System1"
        name2 = sys.argv[4] if len(sys.argv) > 4 else "System2"
        
        print(f"Loading results from {dir1}...")
        data1 = load_results(dir1)
        
        print(f"Loading results from {dir2}...")
        data2 = load_results(dir2)
        
        if not data1 or not data2:
            print("Error: Could not load results from one or both directories")
            sys.exit(1)
        
        print(f"{name1}: {len(data1)} benchmarks")
        print(f"{name2}: {len(data2)} benchmarks")
        
        # Create comparison output directory
        output_dir = f"pagefault_comparison_{name1.replace(' ', '_')}_vs_{name2.replace(' ', '_')}"
        print(f"Generating comparison plots in {output_dir}...")
        
        plot_all_comparisons(data1, data2, output_dir, name1, name2)
        plot_summary_bar_chart(data1, data2, output_dir, name1, name2)
        
        print(f"\nDone! Comparison plots saved to {output_dir}")


if __name__ == '__main__':
    main()
