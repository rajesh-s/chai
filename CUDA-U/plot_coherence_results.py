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

# Colors for single system plot
COLORS_SINGLE = plt.cm.viridis(np.linspace(0.1, 0.9, 7))

# Colors for comparison (two systems)
COLOR_SYSTEM1 = '#1f77b4'  # Blue for GH200
COLOR_SYSTEM2 = '#d62728'  # Red for x86+H100


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
                       marker=MARKERS[i % len(MARKERS)] if 'MARKERS' in dir() else 'o',
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


def plot_comparison_fig3_style(df1, df2, bench_name, output_dir, name1="GH200", name2="x86+H100"):
    """
    Compare two systems in Figure 3 style.
    Same partition values use same marker shapes.
    Different systems use different colors.
    X-axis: GPU (blocks) + CPU (threads) configuration
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Check if we have gpu_blocks column
    has_gpu_blocks = 'gpu_blocks' in df1.columns and df1['gpu_blocks'].notna().any()
    
    # Create configuration labels
    if has_gpu_blocks:
        df1 = df1.copy()
        df2 = df2.copy()
        df1['config'] = df1.apply(lambda r: f"GPU({int(r['gpu_blocks'])}) + CPU({int(r['threads'])})", axis=1)
        df2['config'] = df2.apply(lambda r: f"GPU({int(r['gpu_blocks'])}) + CPU({int(r['threads'])})", axis=1)
        df1['config_order'] = df1['gpu_blocks'] * 1000 + df1['threads']
        df2['config_order'] = df2['gpu_blocks'] * 1000 + df2['threads']
        configs = sorted(set(df1['config'].unique()) | set(df2['config'].unique()), 
                        key=lambda c: df1[df1['config']==c]['config_order'].iloc[0] if len(df1[df1['config']==c]) > 0 else 0)
        config_to_x = {c: i for i, c in enumerate(configs)}
        df1['x_pos'] = df1['config'].map(config_to_x)
        df2['x_pos'] = df2['config'].map(config_to_x)
    else:
        df1 = df1.copy()
        df2 = df2.copy()
        threads = sorted(set(df1['threads'].unique()) | set(df2['threads'].unique()))
        df1['config'] = df1['threads'].apply(lambda t: f"CPU({int(t)})")
        df2['config'] = df2['threads'].apply(lambda t: f"CPU({int(t)})")
        configs = [f"CPU({int(t)})" for t in threads]
        config_to_x = {c: i for i, c in enumerate(configs)}
        df1['x_pos'] = df1['config'].map(config_to_x)
        df2['x_pos'] = df2['config'].map(config_to_x)
    
    if 'partition' not in df1.columns or df1['partition'].isna().all():
        # Task-based benchmark - just compare by config
        sub1 = df1.sort_values('x_pos').groupby('x_pos').first().reset_index()
        sub2 = df2.sort_values('x_pos').groupby('x_pos').first().reset_index()
        
        ax.plot(sub1['x_pos'], sub1['time_ms'],
               marker='o', color=COLOR_SYSTEM1,
               linewidth=2, markersize=8, label=name1)
        ax.plot(sub2['x_pos'], sub2['time_ms'],
               marker='o', color=COLOR_SYSTEM2, linestyle='--',
               linewidth=2, markersize=8, label=name2)
    else:
        partitions = sorted(set(df1['partition'].dropna().unique()) & 
                           set(df2['partition'].dropna().unique()))
        
        for p in partitions:
            style = PARTITION_STYLES.get(p, {'marker': 'o', 'linestyle': '-', 'label': f'α={p}'})
            
            # System 1 (solid lines)
            sub1 = df1[df1['partition'] == p].sort_values('x_pos')
            if len(sub1) > 0:
                ax.plot(sub1['x_pos'], sub1['time_ms'],
                       marker=style['marker'],
                       linestyle='-',
                       color=COLOR_SYSTEM1,
                       linewidth=2, markersize=8,
                       label=f'{name1} {style["label"]}')
            
            # System 2 (dashed lines)
            sub2 = df2[df2['partition'] == p].sort_values('x_pos')
            if len(sub2) > 0:
                ax.plot(sub2['x_pos'], sub2['time_ms'],
                       marker=style['marker'],
                       linestyle='--',
                       color=COLOR_SYSTEM2,
                       linewidth=2, markersize=8,
                       label=f'{name2} {style["label"]}')
    
    # X-axis
    ax.set_xlabel('Configuration: GPU (blocks) + CPU (threads)', fontsize=12)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    
    # Y-axis
    ax.set_ylabel('Kernel Time (ms)', fontsize=12)
    
    # Title
    ax.set_title(f'{bench_name}\n{name1} (solid, blue) vs {name2} (dashed, red)', 
                fontsize=14, fontweight='bold')
    
    # Legend - outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{bench_name}_comparison.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{bench_name}_comparison.pdf'), bbox_inches='tight')
    plt.close()


def plot_speedup_comparison(df1, df2, bench_name, output_dir, name1="GH200", name2="x86+H100"):
    """
    Plot speedup of System1 over System2.
    Shows where HW coherence wins.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'partition' not in df1.columns or df1['partition'].isna().all():
        return
    
    partitions = sorted(set(df1['partition'].dropna().unique()) & 
                       set(df2['partition'].dropna().unique()))
    
    # Skip 0.0 (GPU only) and 1.0 (CPU only) for speedup - those are references
    collab_partitions = [p for p in partitions if 0 < p < 1]
    
    for i, p in enumerate(collab_partitions):
        sub1 = df1[df1['partition'] == p].set_index('threads')['time_ms']
        sub2 = df2[df2['partition'] == p].set_index('threads')['time_ms']
        
        common_threads = sorted(set(sub1.index) & set(sub2.index))
        if len(common_threads) == 0:
            continue
        
        speedup = sub2.loc[common_threads] / sub1.loc[common_threads]
        
        style = PARTITION_STYLES.get(p, {'marker': 'o', 'label': f'α={p}'})
        ax.plot(common_threads, speedup,
               marker=style['marker'],
               color=COLORS_SINGLE[i % len(COLORS_SINGLE)],
               linewidth=2, markersize=8,
               label=style['label'])
    
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='No speedup')
    
    ax.set_xlabel('CPU Threads', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_ylabel(f'Speedup ({name1} / {name2})', fontsize=12)
    ax.set_title(f'{bench_name}\nHW Coherence Advantage', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{bench_name}_speedup.png'), dpi=150, bbox_inches='tight')
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
