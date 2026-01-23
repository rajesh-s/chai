#!/usr/bin/env python3
"""
Plot Coherence Study Results - Figure 3 Style from ISPASS'17 Paper
X-axis: CPU/GPU configurations (threads)
Y-axis: Execution time (ms)
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
    csv_files = glob.glob(os.path.join(results_dir, "*_results.csv"))
    
    for csv_file in csv_files:
        bench_name = os.path.basename(csv_file).replace("_results.csv", "")
        df = pd.read_csv(csv_file)
        
        # Convert partition to numeric (handle 'dynamic' for task-based benchmarks)
        if 'partition' in df.columns:
            df['partition'] = pd.to_numeric(df['partition'], errors='coerce')
        
        # Handle switching_limit for BFS/SSSP
        if 'switching_limit' in df.columns:
            df['switching_limit'] = pd.to_numeric(df['switching_limit'], errors='coerce')
        
        # Use kernel_time_ms if available
        if 'kernel_time_ms' in df.columns:
            df['time_ms'] = pd.to_numeric(df['kernel_time_ms'], errors='coerce')
        
        data[bench_name] = df
    
    return data


def plot_benchmark_fig3_style(df, bench_name, output_dir, system_name="System"):
    """
    Plot single benchmark in Figure 3 style.
    Creates separate subplots for each GPU block count.
    X-axis: CPU threads
    Y-axis: Execution time (ms)
    Different markers/lines for each partition ratio (α)
    """
    
    # Check if we have numeric gpu_blocks column (not 'default')
    has_gpu_blocks = False
    if 'gpu_blocks' in df.columns:
        # Try to convert to numeric, treating 'default' as NaN
        numeric_gpu = pd.to_numeric(df['gpu_blocks'], errors='coerce')
        has_gpu_blocks = numeric_gpu.notna().any()
        if has_gpu_blocks:
            df = df.copy()
            df['gpu_blocks'] = numeric_gpu
    
    # Check if this is a partition-based benchmark
    has_partitions = 'partition' in df.columns and df['partition'].notna().any()
    
    if has_gpu_blocks and has_partitions:
        # Create subplots - one per GPU block count
        gpu_blocks_vals = sorted(df['gpu_blocks'].dropna().unique())
        n_subplots = len(gpu_blocks_vals)
        
        # Determine grid layout
        if n_subplots <= 2:
            nrows, ncols = 1, n_subplots
        elif n_subplots <= 4:
            nrows, ncols = 2, 2
        elif n_subplots <= 6:
            nrows, ncols = 2, 3
        else:
            nrows, ncols = 3, 3
        
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
                
                # Get style for this partition
                style = PARTITION_STYLES.get(p, {'marker': 'o', 'linestyle': '-', 'label': f'α={p}'})
                
                ax.plot(subset['threads'], subset['time_ms'],
                       marker=style['marker'],
                       linestyle=style['linestyle'],
                       color=COLORS_SINGLE[i % len(COLORS_SINGLE)],
                       linewidth=2, markersize=8,
                       label=style['label'])
            
            ax.set_xlabel('CPU Threads', fontsize=10)
            ax.set_ylabel('Kernel Time (ms)', fontsize=10)
            ax.set_title(f'GPU Blocks = {int(g)}', fontsize=11, fontweight='bold')
            ax.set_xticks(threads)
            ax.grid(True, alpha=0.3)
            
            # Only show legend on first subplot
            if idx == 0:
                ax.legend(title='Partition (α)', loc='best', fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(gpu_blocks_vals), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{bench_name} - {system_name}', fontsize=14, fontweight='bold')
        
    elif has_partitions:
        # No GPU blocks variation - single plot with threads on x-axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        partitions = sorted(df['partition'].dropna().unique())
        threads = sorted(df['threads'].unique())
        
        for i, p in enumerate(partitions):
            subset = df[df['partition'] == p].sort_values('threads')
            
            if len(subset) == 0:
                continue
            
            style = PARTITION_STYLES.get(p, {'marker': 'o', 'linestyle': '-', 'label': f'α={p}'})
            
            ax.plot(subset['threads'], subset['time_ms'],
                   marker=style['marker'],
                   linestyle=style['linestyle'],
                   color=COLORS_SINGLE[i % len(COLORS_SINGLE)],
                   linewidth=2, markersize=8,
                   label=style['label'])
        
        ax.set_xlabel('CPU Threads', fontsize=12)
        ax.set_ylabel('Kernel Time (ms)', fontsize=12)
        ax.set_title(f'{bench_name} - {system_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(threads)
        ax.legend(title='Partition (α)', loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
    else:
        # Dynamic/task-based benchmark - subplots by GPU blocks if available
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if has_gpu_blocks:
            gpu_blocks_vals = sorted(df['gpu_blocks'].dropna().unique())
            threads = sorted(df['threads'].unique())
            
            for i, g in enumerate(gpu_blocks_vals):
                subset = df[df['gpu_blocks'] == g].sort_values('threads')
                ax.plot(subset['threads'], subset['time_ms'],
                       marker=MARKERS[i % len(MARKERS)],
                       color=COLORS_SINGLE[i % len(COLORS_SINGLE)],
                       linewidth=2, markersize=8,
                       label=f'GPU({int(g)} blocks)')
            
            ax.set_xticks(threads)
            ax.legend(title='GPU Blocks', loc='best', fontsize=9)
        else:
            threads = sorted(df['threads'].unique())
            subset = df.sort_values('threads')
            ax.plot(subset['threads'], subset['time_ms'],
                   marker='o', color=COLORS_SINGLE[0],
                   linewidth=2, markersize=8,
                   label='Dynamic')
            ax.set_xticks(threads)
            ax.legend(loc='best')
        
        ax.set_xlabel('CPU Threads', fontsize=12)
        ax.set_ylabel('Kernel Time (ms)', fontsize=12)
        ax.set_title(f'{bench_name} - {system_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{bench_name}_fig3.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{bench_name}_fig3.pdf'), bbox_inches='tight')
    plt.close()
    
    return True


def plot_all_benchmarks(data, output_dir, system_name="System"):
    """Generate Figure 3 style plots for all benchmarks."""
    os.makedirs(output_dir, exist_ok=True)
    
    for bench_name, df in data.items():
        print(f"  Plotting {bench_name}...")
        plot_benchmark_fig3_style(df, bench_name, output_dir, system_name)
        print(f"    Saved {bench_name}_fig3.png")


def _create_grouped_legend(ax, handles1, labels1, handles2, labels2, name1, name2, loc='best'):
    """Create a grouped legend with system titles."""
    from matplotlib.lines import Line2D
    
    # Create header entries (invisible lines with bold labels)
    header1 = Line2D([0], [0], color='none', label=f'▬ {name1} (solid)')
    header2 = Line2D([0], [0], color='none', label=f'┄ {name2} (dashed)')
    
    # Combine: header1, items1, spacer, header2, items2
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


def plot_comparison_fig3_style(df1, df2, bench_name, output_dir, name1="GH200", name2="x86+H100"):
    """
    Compare two systems in Figure 3 style (matching standalone plot format).
    X-axis: CPU threads
    Subplots: GPU block counts (if applicable)
    Two color palettes: COLORS_SYSTEM1 for system1, COLORS_SYSTEM2 for system2
    Same markers for same partition values across both systems.
    """
    
    # Check if we have numeric gpu_blocks column (not 'default')
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
    
    # Check if this is a partition-based benchmark
    has_partitions = 'partition' in df1.columns and df1['partition'].notna().any()
    
    if has_gpu_blocks and has_partitions:
        # Create subplots - one per GPU block count
        gpu_blocks_vals = sorted(set(df1['gpu_blocks'].dropna().unique()) | 
                                  set(df2['gpu_blocks'].dropna().unique()))
        n_subplots = len(gpu_blocks_vals)
        
        # Determine grid layout
        if n_subplots <= 2:
            nrows, ncols = 1, n_subplots
        elif n_subplots <= 4:
            nrows, ncols = 2, 2
        elif n_subplots <= 6:
            nrows, ncols = 2, 3
        else:
            nrows, ncols = 3, 3
        
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
                
                # System 1 (solid lines)
                subset1 = subset_g1[subset_g1['partition'] == p].sort_values('threads')
                if len(subset1) > 0:
                    line1, = ax.plot(subset1['threads'], subset1['time_ms'],
                           marker=style['marker'],
                           linestyle='-',
                           color=COLORS_SYSTEM1[i % len(COLORS_SYSTEM1)],
                           linewidth=2, markersize=8)
                    handles1.append(line1)
                    labels1.append(style['label'])
                
                # System 2 (dashed lines)
                subset2 = subset_g2[subset_g2['partition'] == p].sort_values('threads')
                if len(subset2) > 0:
                    line2, = ax.plot(subset2['threads'], subset2['time_ms'],
                           marker=style['marker'],
                           linestyle='--',
                           color=COLORS_SYSTEM2[i % len(COLORS_SYSTEM2)],
                           linewidth=2, markersize=8)
                    handles2.append(line2)
                    labels2.append(style['label'])
            
            ax.set_xlabel('CPU Threads', fontsize=10)
            ax.set_ylabel('Kernel Time (ms)', fontsize=10)
            ax.set_title(f'GPU Blocks = {int(g)}', fontsize=11, fontweight='bold')
            ax.set_xticks(threads)
            ax.grid(True, alpha=0.3)
            
            # Store handles from first subplot for legends
            if idx == 0:
                first_handles1, first_labels1 = handles1.copy(), labels1.copy()
                first_handles2, first_labels2 = handles2.copy(), labels2.copy()
        
        # Put system 1 legend on first subplot, system 2 legend on second subplot
        _create_single_legend(axes[0], first_handles1, first_labels1, name1, '▬', loc='upper right')
        if len(gpu_blocks_vals) > 1:
            _create_single_legend(axes[1], first_handles2, first_labels2, name2, '┄', loc='upper right')
        
        # Hide unused subplots
        for idx in range(len(gpu_blocks_vals), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{bench_name}', fontsize=14, fontweight='bold')
        
    elif has_partitions:
        # No GPU blocks variation - single plot with threads on x-axis
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
            
            # System 1 (solid lines)
            subset1 = df1[df1['partition'] == p].sort_values('threads')
            if len(subset1) > 0:
                line1, = ax.plot(subset1['threads'], subset1['time_ms'],
                       marker=style['marker'],
                       linestyle='-',
                       color=COLORS_SYSTEM1[i % len(COLORS_SYSTEM1)],
                       linewidth=2, markersize=8)
                handles1.append(line1)
                labels1.append(style['label'])
            
            # System 2 (dashed lines)
            subset2 = df2[df2['partition'] == p].sort_values('threads')
            if len(subset2) > 0:
                line2, = ax.plot(subset2['threads'], subset2['time_ms'],
                       marker=style['marker'],
                       linestyle='--',
                       color=COLORS_SYSTEM2[i % len(COLORS_SYSTEM2)],
                       linewidth=2, markersize=8)
                handles2.append(line2)
                labels2.append(style['label'])
        
        ax.set_xlabel('CPU Threads', fontsize=12)
        ax.set_ylabel('Kernel Time (ms)', fontsize=12)
        ax.set_title(f'{bench_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(threads)
        _create_grouped_legend(ax, handles1, labels1, handles2, labels2, name1, name2, loc='upper left')
        ax.grid(True, alpha=0.3)
        
    else:
        # Dynamic/task-based benchmark - compare by GPU blocks or just threads
        # Use more compact figure size for better appearance in presentations
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
                # System 1 (solid lines)
                subset1 = df1[df1['gpu_blocks'] == g].sort_values('threads')
                if len(subset1) > 0:
                    line1, = ax.plot(subset1['threads'], subset1['time_ms'],
                           marker=MARKERS[i % len(MARKERS)],
                           linestyle='-',
                           color=COLORS_SYSTEM1[i % len(COLORS_SYSTEM1)],
                           linewidth=2.5, markersize=10)
                    handles1.append(line1)
                    labels1.append(f'GPU ({int(g)} blocks)')
                
                # System 2 (dashed lines)
                subset2 = df2[df2['gpu_blocks'] == g].sort_values('threads')
                if len(subset2) > 0:
                    line2, = ax.plot(subset2['threads'], subset2['time_ms'],
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
            
            line1, = ax.plot(subset1['threads'], subset1['time_ms'],
                   marker='o', linestyle='-',
                   color=COLORS_SYSTEM1[3],
                   linewidth=2.5, markersize=10)
            line2, = ax.plot(subset2['threads'], subset2['time_ms'],
                   marker='o', linestyle='--',
                   color=COLORS_SYSTEM2[3],
                   linewidth=2.5, markersize=10)
            
            ax.set_xticks(threads)
            _create_grouped_legend(ax, [line1], ['Dynamic'], [line2], ['Dynamic'], name1, name2, loc='upper left')
        
        ax.set_xlabel('CPU Threads', fontsize=14)
        ax.set_ylabel('Kernel Time (ms)', fontsize=14)
        ax.set_title(f'{bench_name}', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{bench_name}_comparison.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{bench_name}_comparison.pdf'), bbox_inches='tight')
    plt.close()


def plot_speedup_comparison(df1, df2, bench_name, output_dir, name1="GH200", name2="x86+H100"):
    """
    Plot speedup of System1 over System2.
    Shows where HW coherence wins (speedup > 1).
    Matches the style of comparison plots.
    """
    
    # Check if we have numeric gpu_blocks
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
    
    # Check if partition-based
    has_partitions = 'partition' in df1.columns and df1['partition'].notna().any()
    
    if has_gpu_blocks and has_partitions:
        # Subplots for each GPU block count
        gpu_blocks_vals = sorted(set(df1['gpu_blocks'].dropna().unique()) & 
                                  set(df2['gpu_blocks'].dropna().unique()))
        n_subplots = len(gpu_blocks_vals)
        
        if n_subplots == 0:
            return
        
        if n_subplots <= 2:
            nrows, ncols = 1, n_subplots
        elif n_subplots <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), squeeze=False)
        axes = axes.flatten()
        
        partitions = sorted(set(df1['partition'].dropna().unique()) & 
                           set(df2['partition'].dropna().unique()))
        # Include all partitions including 0.0 and 1.0
        
        for idx, g in enumerate(gpu_blocks_vals):
            ax = axes[idx]
            subset_g1 = df1[df1['gpu_blocks'] == g]
            subset_g2 = df2[df2['gpu_blocks'] == g]
            
            all_threads = set()
            for i, p in enumerate(partitions):
                style = PARTITION_STYLES.get(p, {'marker': 'o', 'linestyle': '-', 'label': f'α={p}'})
                
                sub1 = subset_g1[subset_g1['partition'] == p].groupby('threads')['time_ms'].mean()
                sub2 = subset_g2[subset_g2['partition'] == p].groupby('threads')['time_ms'].mean()
                
                common_threads = sorted(set(sub1.index) & set(sub2.index))
                if len(common_threads) == 0:
                    continue
                
                all_threads.update(common_threads)
                speedup = sub2.loc[common_threads] / sub1.loc[common_threads]
                
                ax.plot(common_threads, speedup.values,
                       marker=style['marker'],
                       linestyle='-',
                       color=COLORS_SINGLE[i % len(COLORS_SINGLE)],
                       linewidth=2.5, markersize=10,
                       label=style['label'])
            
            ax.set_ylim(bottom=1)
            ax.set_xlabel('CPU Threads', fontsize=10)
            ax.set_ylabel('Speedup', fontsize=10)
            ax.set_title(f'GPU ({int(g)} blocks)', fontsize=11, fontweight='bold')
            ax.set_xticks(sorted(all_threads))
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        for idx in range(len(gpu_blocks_vals), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{bench_name} - Speedup ({name1} over {name2})', fontsize=14, fontweight='bold')
        
    elif has_partitions:
        # Single plot with partitions
        fig, ax = plt.subplots(figsize=(8, 6))
        
        df1 = df1.copy()
        df2 = df2.copy()
        
        partitions = sorted(set(df1['partition'].dropna().unique()) & 
                           set(df2['partition'].dropna().unique()))
        
        all_threads = set()
        for i, p in enumerate(partitions):
            style = PARTITION_STYLES.get(p, {'marker': 'o', 'linestyle': '-', 'label': f'α={p}'})
            
            sub1 = df1[df1['partition'] == p].groupby('threads')['time_ms'].mean()
            sub2 = df2[df2['partition'] == p].groupby('threads')['time_ms'].mean()
            
            common_threads = sorted(set(sub1.index) & set(sub2.index))
            if len(common_threads) == 0:
                continue
            
            all_threads.update(common_threads)
            speedup = sub2.loc[common_threads] / sub1.loc[common_threads]
            
            ax.plot(common_threads, speedup.values,
                   marker=style['marker'],
                   linestyle='-',
                   color=COLORS_SINGLE[i % len(COLORS_SINGLE)],
                   linewidth=2.5, markersize=10,
                   label=style['label'])
        
        ax.set_ylim(bottom=1)
        ax.set_xlabel('CPU Threads', fontsize=14)
        ax.set_ylabel('Speedup', fontsize=14)
        ax.set_title(f'{bench_name} - Speedup ({name1} over {name2})', fontsize=16, fontweight='bold')
        ax.set_xticks(sorted(all_threads))
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(title='Partition (α)', loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
    else:
        # Dynamic benchmark - speedup by GPU blocks
        fig, ax = plt.subplots(figsize=(8, 6))
        
        df1 = df1.copy()
        df2 = df2.copy()
        
        all_threads = set()
        if has_gpu_blocks:
            gpu_blocks_vals = sorted(set(df1['gpu_blocks'].dropna().unique()) & 
                                      set(df2['gpu_blocks'].dropna().unique()))
            
            for i, g in enumerate(gpu_blocks_vals):
                sub1 = df1[df1['gpu_blocks'] == g].groupby('threads')['time_ms'].mean()
                sub2 = df2[df2['gpu_blocks'] == g].groupby('threads')['time_ms'].mean()
                
                common_threads = sorted(set(sub1.index) & set(sub2.index))
                if len(common_threads) == 0:
                    continue
                
                all_threads.update(common_threads)
                speedup = sub2.loc[common_threads] / sub1.loc[common_threads]
                
                ax.plot(common_threads, speedup.values,
                       marker=MARKERS[i % len(MARKERS)],
                       linestyle='-',
                       color=COLORS_SINGLE[i % len(COLORS_SINGLE)],
                       linewidth=2.5, markersize=10,
                       label=f'GPU ({int(g)} blocks)')
            
            ax.legend(title='GPU Blocks', loc='upper left', fontsize=10)
        else:
            sub1 = df1.groupby('threads')['time_ms'].mean()
            sub2 = df2.groupby('threads')['time_ms'].mean()
            
            common_threads = sorted(set(sub1.index) & set(sub2.index))
            if len(common_threads) > 0:
                all_threads.update(common_threads)
                speedup = sub2.loc[common_threads] / sub1.loc[common_threads]
                ax.plot(common_threads, speedup.values,
                       marker='o', linestyle='-',
                       color=COLORS_SINGLE[3],
                       linewidth=2.5, markersize=10,
                       label='Dynamic')
                ax.legend(loc='upper left', fontsize=10)
        
        ax.set_ylim(bottom=1)
        ax.set_xlabel('CPU Threads', fontsize=14)
        ax.set_ylabel('Speedup', fontsize=14)
        ax.set_title(f'{bench_name} - Speedup ({name1} over {name2})', fontsize=16, fontweight='bold')
        ax.set_xticks(sorted(all_threads))
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{bench_name}_speedup.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{bench_name}_speedup.pdf'), bbox_inches='tight')
    plt.close()


def plot_summary_heatmap(data, output_dir, system_name="System"):
    """
    Create summary heatmap for benchmarks with partitioning.
    """
    # Filter to partition-based benchmarks
    partition_benchmarks = {k: v for k, v in data.items() 
                           if 'partition' in v.columns and v['partition'].notna().any()}
    
    if not partition_benchmarks:
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, (bench_name, df) in enumerate(partition_benchmarks.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Create config label combining gpu_blocks and threads
        df = df.copy()
        
        # Convert gpu_blocks to numeric, treating 'default' as NaN
        has_gpu_blocks = False
        if 'gpu_blocks' in df.columns:
            numeric_gpu = pd.to_numeric(df['gpu_blocks'], errors='coerce')
            has_gpu_blocks = numeric_gpu.notna().any()
            if has_gpu_blocks:
                df['gpu_blocks'] = numeric_gpu
        
        if has_gpu_blocks:
            df['config'] = df.apply(lambda r: f"G{int(r['gpu_blocks'])}+C{int(r['threads'])}", axis=1)
            df['config_order'] = df['gpu_blocks'] * 1000 + df['threads']
            df = df.sort_values('config_order')
        else:
            df['config'] = df['threads'].apply(lambda t: f"C{int(t)}")
        
        # Pivot table with config on y-axis
        pivot = df.pivot_table(index='config', columns='partition', values='time_ms', aggfunc='mean')
        
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r')
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{p:.2f}' for p in pivot.columns], fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=6)
        
        ax.set_xlabel('Partition (α)', fontsize=9)
        ax.set_ylabel('GPU(G) + CPU(C)', fontsize=9)
        ax.set_title(bench_name, fontsize=11, fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide unused axes
    for idx in range(len(partition_benchmarks), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Kernel Time Heatmaps - {system_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_heatmaps.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'summary_heatmaps.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved summary_heatmaps.png")


def compare_systems(dir1, dir2, output_dir, name1="GH200", name2="x86+H100"):
    """Compare results from two different systems."""
    data1 = load_results(dir1)
    data2 = load_results(dir2)
    
    os.makedirs(output_dir, exist_ok=True)
    
    common_benchmarks = set(data1.keys()) & set(data2.keys())
    
    print(f"\nComparing {len(common_benchmarks)} benchmarks:")
    
    for bench_name in sorted(common_benchmarks):
        print(f"  {bench_name}...")
        df1 = data1[bench_name]
        df2 = data2[bench_name]
        
        plot_comparison_fig3_style(df1, df2, bench_name, output_dir, name1, name2)
        print(f"    Saved {bench_name}_comparison.png")
        
        plot_speedup_comparison(df1, df2, bench_name, output_dir, name1, name2)
        print(f"    Saved {bench_name}_speedup.png")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single system:  python plot_coherence_results.py <results_dir> [system_name]")
        print("  Compare two:    python plot_coherence_results.py --compare <dir1> <dir2> [name1] [name2]")
        print()
        print("Example:")
        print("  python plot_coherence_results.py results_gh200_20260121/ 'GH200 (HW Coherence)'")
        print("  python plot_coherence_results.py --compare results_gh200/ results_h100/ GH200 'x86+H100'")
        sys.exit(1)
    
    if sys.argv[1] == "--compare":
        if len(sys.argv) < 4:
            print("Error: --compare requires two result directories")
            sys.exit(1)
        
        dir1 = sys.argv[2]
        dir2 = sys.argv[3]
        name1 = sys.argv[4] if len(sys.argv) > 4 else "System1"
        name2 = sys.argv[5] if len(sys.argv) > 5 else "System2"
        
        output_dir = "comparison_plots"
        
        print(f"Comparing {name1} vs {name2}...")
        compare_systems(dir1, dir2, output_dir, name1, name2)
        print(f"\nComparison plots saved to: {output_dir}/")
    else:
        results_dir = sys.argv[1]
        system_name = sys.argv[2] if len(sys.argv) > 2 else Path(results_dir).name
        
        if not os.path.isdir(results_dir):
            print(f"Error: Directory not found: {results_dir}")
            sys.exit(1)
        
        print(f"Loading results from: {results_dir}")
        data = load_results(results_dir)
        
        if not data:
            print("Error: No CSV files found in directory")
            sys.exit(1)
        
        print(f"Found benchmarks: {list(data.keys())}")
        
        plots_dir = os.path.join(results_dir, "plots")
        
        print(f"\nGenerating Figure 3 style plots...")
        plot_all_benchmarks(data, plots_dir, system_name)
        
        print(f"\nGenerating summary heatmaps...")
        plot_summary_heatmap(data, plots_dir, system_name)
        
        print(f"\nAll plots saved to: {plots_dir}/")


# Markers list for fallback
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

if __name__ == "__main__":
    main()
