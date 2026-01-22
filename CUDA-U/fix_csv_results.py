#!/usr/bin/env python3
"""
Fix malformed CSV results by re-parsing raw log files.
The issue is that benchmarks with CPU/GPU proxies have multiple "Kernel Time" lines,
and the correct time to use is "Total Proxies Time".
"""

import os
import re
import sys
from pathlib import Path

def parse_raw_log(log_file, bench_name):
    """Parse a raw log file and extract timing data."""
    results = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Split by run markers
    runs = re.split(r'---\s*(.*?)\s*---', content)
    
    i = 1
    while i < len(runs):
        config_str = runs[i].strip()
        output = runs[i+1] if i+1 < len(runs) else ""
        i += 2
        
        # Parse config: "threads=X gpu_blocks=Y partition=Z" or "threads=X partition=Z"
        threads = re.search(r'threads=(\d+)', config_str)
        gpu_blocks = re.search(r'gpu_blocks=(\d+)', config_str)
        partition = re.search(r'partition=([\d.]+|dynamic)', config_str)
        switching_limit = re.search(r'switching_limit=(\d+)', config_str)
        
        threads = int(threads.group(1)) if threads else 1
        gpu_blocks = int(gpu_blocks.group(1)) if gpu_blocks else "default"
        partition = partition.group(1) if partition else "dynamic"
        
        # Extract timing data - prefer "Total Proxies Time" over individual Kernel Times
        alloc_match = re.search(r'Allocation Time \(ms\):\s*([\d.]+)', output)
        init_match = re.search(r'Initialization Time \(ms\):\s*([\d.]+)', output)
        dealloc_match = re.search(r'Deallocation Time \(ms\):\s*([\d.]+)', output)
        
        # For kernel time, prioritize:
        # 1. "Total Proxies Time" (for CPU/GPU cooperative benchmarks)
        # 2. First "Kernel Time" that's not a sub-component (not indented/prefixed with proxy name)
        # 3. Any "Kernel Time"
        
        kernel_time = None
        
        # Try Total Proxies Time first
        total_proxies = re.search(r'Total Proxies Time \(ms\):\s*([\d.]+)', output)
        if total_proxies:
            kernel_time = float(total_proxies.group(1))
        
        # If no Total Proxies Time, look for standalone Kernel Time
        if kernel_time is None:
            # Match "Kernel Time" at start of line (not indented/prefixed)
            standalone_kernel = re.search(r'^Kernel Time \(ms\):\s*([\d.]+)', output, re.MULTILINE)
            if standalone_kernel:
                kernel_time = float(standalone_kernel.group(1))
        
        # Fallback: any Kernel Time
        if kernel_time is None:
            any_kernel = re.search(r'Kernel Time \(ms\):\s*([\d.]+)', output)
            if any_kernel:
                kernel_time = float(any_kernel.group(1))
        
        if kernel_time is None:
            kernel_time = "NA"
        
        alloc_time = float(alloc_match.group(1)) if alloc_match else "NA"
        init_time = float(init_match.group(1)) if init_match else "NA"
        dealloc_time = float(dealloc_match.group(1)) if dealloc_match else "NA"
        
        # Check for TIMEOUT
        if "TIMEOUT" in output or "timeout" in config_str.lower():
            kernel_time = "TIMEOUT"
            alloc_time = "TIMEOUT"
            init_time = "TIMEOUT"
            dealloc_time = "TIMEOUT"
        
        # Add switching_limit if present (for BFS/SSSP)
        extra_cols = {}
        if switching_limit:
            extra_cols['switching_limit'] = int(switching_limit.group(1))
        
        results.append({
            'benchmark': bench_name,
            'threads': threads,
            'gpu_blocks': gpu_blocks,
            'partition': partition,
            'kernel_time_ms': kernel_time,
            'alloc_time_ms': alloc_time,
            'init_time_ms': init_time,
            'dealloc_time_ms': dealloc_time,
            **extra_cols
        })
    
    return results


def fix_results_dir(results_dir):
    """Re-parse all raw logs and regenerate CSV files."""
    results_path = Path(results_dir)
    
    raw_logs = list(results_path.glob("*_raw.log"))
    
    if not raw_logs:
        print(f"No raw log files found in {results_dir}")
        return
    
    print(f"Found {len(raw_logs)} raw log files")
    
    for log_file in raw_logs:
        bench_name = log_file.stem.replace("_raw", "")
        csv_file = results_path / f"{bench_name}_results.csv"
        
        print(f"  Processing {bench_name}...")
        
        results = parse_raw_log(log_file, bench_name)
        
        if not results:
            print(f"    WARNING: No results parsed from {log_file}")
            continue
        
        # Determine columns (some benchmarks have switching_limit)
        has_switching_limit = any('switching_limit' in r for r in results)
        
        # Write CSV
        with open(csv_file, 'w') as f:
            if has_switching_limit:
                f.write("benchmark,threads,gpu_blocks,partition,switching_limit,kernel_time_ms,alloc_time_ms,init_time_ms,dealloc_time_ms\n")
                for r in results:
                    sl = r.get('switching_limit', 'NA')
                    f.write(f"{r['benchmark']},{r['threads']},{r['gpu_blocks']},{r['partition']},{sl},{r['kernel_time_ms']},{r['alloc_time_ms']},{r['init_time_ms']},{r['dealloc_time_ms']}\n")
            else:
                f.write("benchmark,threads,gpu_blocks,partition,kernel_time_ms,alloc_time_ms,init_time_ms,dealloc_time_ms\n")
                for r in results:
                    f.write(f"{r['benchmark']},{r['threads']},{r['gpu_blocks']},{r['partition']},{r['kernel_time_ms']},{r['alloc_time_ms']},{r['init_time_ms']},{r['dealloc_time_ms']}\n")
        
        print(f"    Wrote {len(results)} rows to {csv_file.name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_csv_results.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    fix_results_dir(results_dir)
    print("Done!")


if __name__ == "__main__":
    main()
