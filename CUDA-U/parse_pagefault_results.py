#!/usr/bin/env python3
"""
Parse nsys page fault reports and create summary CSVs.
Processes raw nsys-rep files to extract CPU and GPU page fault counts and data sizes.
"""

import os
import re
import sys
import glob
import subprocess
from pathlib import Path
import csv


def parse_nsys_report(nsys_file):
    """
    Parse an nsys-rep file and extract page fault statistics.
    Returns dict with cpu_page_faults, gpu_page_faults, cpu_pf_data_mb, gpu_pf_data_mb
    """
    result = {
        'cpu_page_faults': 0,
        'gpu_page_faults': 0,
        'htod_migration_mb': 0,
        'dtoh_migration_mb': 0,
    }
    
    if not os.path.exists(nsys_file):
        return None
    
    # Use nsys stats with um_total_sum report (CSV format for easy parsing)
    try:
        output = subprocess.run(
            ['nsys', 'stats', '--report', 'um_total_sum', 
             '--format', 'csv', '--force-export=true', nsys_file],
            capture_output=True, text=True, timeout=120
        )
        
        if output.returncode == 0 and output.stdout.strip():
            lines = output.stdout.strip().split('\n')
            # Find the CSV data line (skip headers and notices)
            for line in lines:
                # CSV line has commas and numbers
                if ',' in line and not line.startswith(' ') and not line.startswith('NOTICE') and not line.startswith('Processing'):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        try:
                            # Format: HtoD MB, DtoH MB, CPU PF, GPU PF, ...
                            result['htod_migration_mb'] = float(parts[0]) if parts[0] else 0
                            result['dtoh_migration_mb'] = float(parts[1]) if parts[1] else 0
                            result['cpu_page_faults'] = int(parts[2]) if parts[2] else 0
                            result['gpu_page_faults'] = int(parts[3]) if parts[3] else 0
                            break
                        except (ValueError, IndexError):
                            pass
                            
    except subprocess.TimeoutExpired:
        print(f"  Warning: Timeout parsing {nsys_file}")
        return None
    except FileNotFoundError:
        print("  Error: nsys command not found")
        return None
    except Exception as e:
        print(f"  Warning: Error parsing {nsys_file}: {e}")
        return None
    
    # Calculate total migration as proxy for page fault data
    result['cpu_pf_data_mb'] = result['dtoh_migration_mb']  # CPU reads from GPU
    result['gpu_pf_data_mb'] = result['htod_migration_mb']  # GPU reads from CPU
    
    return result


def parse_sqlite_report(nsys_file):
    """
    Alternative: Parse nsys SQLite export for page fault data.
    This is used when nsys stats doesn't provide the needed reports.
    """
    import sqlite3
    
    result = {
        'cpu_page_faults': 0,
        'gpu_page_faults': 0,
        'cpu_pf_data_bytes': 0,
        'gpu_pf_data_bytes': 0,
    }
    
    sqlite_file = nsys_file.replace('.nsys-rep', '.sqlite')
    
    # Export to SQLite if needed
    if not os.path.exists(sqlite_file):
        try:
            subprocess.run(
                ['nsys', 'export', '--type', 'sqlite', '--output', sqlite_file, nsys_file],
                capture_output=True, timeout=120
            )
        except Exception as e:
            print(f"  Warning: Could not export to SQLite: {e}")
            return None
    
    if not os.path.exists(sqlite_file):
        return None
    
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Try to get page fault data from various possible tables
        # Table names vary by nsys version
        tables_to_check = [
            'CUDA_UM_CPU_PAGE_FAULTS',
            'CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER',
            'StringIds',  # For checking what's available
        ]
        
        # Check available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        available_tables = [row[0] for row in cursor.fetchall()]
        
        # Try CUDA UM CPU page faults
        if 'CUDA_UM_CPU_PAGE_FAULTS' in available_tables:
            cursor.execute("SELECT COUNT(*), SUM(size) FROM CUDA_UM_CPU_PAGE_FAULTS")
            row = cursor.fetchone()
            if row:
                result['cpu_page_faults'] = row[0] or 0
                result['cpu_pf_data_bytes'] = row[1] or 0
        
        # Try CUDA UM GPU page faults
        if 'CUDA_UM_GPU_PAGE_FAULTS' in available_tables:
            cursor.execute("SELECT COUNT(*), SUM(size) FROM CUDA_UM_GPU_PAGE_FAULTS")
            row = cursor.fetchone()
            if row:
                result['gpu_page_faults'] = row[0] or 0
                result['gpu_pf_data_bytes'] = row[1] or 0
        
        # Try unified memory counter table (older nsys versions)
        if 'CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER' in available_tables:
            # counterKind: 0=bytes_transfer_htod, 1=bytes_transfer_dtoh, 
            # 2=cpu_page_fault_count, 3=gpu_page_fault_count
            cursor.execute("""
                SELECT counterKind, SUM(value) 
                FROM CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER 
                GROUP BY counterKind
            """)
            for row in cursor.fetchall():
                kind, value = row
                if kind == 2:  # CPU page faults
                    result['cpu_page_faults'] = int(value or 0)
                elif kind == 3:  # GPU page faults
                    result['gpu_page_faults'] = int(value or 0)
        
        conn.close()
        
    except Exception as e:
        print(f"  Warning: Error reading SQLite {sqlite_file}: {e}")
        return None
    
    result['cpu_pf_data_mb'] = result['cpu_pf_data_bytes'] / (1024 * 1024)
    result['gpu_pf_data_mb'] = result['gpu_pf_data_bytes'] / (1024 * 1024)
    
    return result


def parse_filename(filename):
    """
    Parse benchmark config from filename.
    Format: BENCH_tTHREADS_gGPUBLOCKS_aPARTITION.nsys-rep
    """
    basename = os.path.basename(filename).replace('.nsys-rep', '')
    
    # Pattern: BENCH_tN_gN_aN.N or BENCH_tN_gdefault_adynamic
    match = re.match(r'(\w+)_t(\d+)_g(\w+)_a([\d.]+|dynamic)', basename)
    
    if match:
        bench = match.group(1)
        threads = int(match.group(2))
        gpu_blocks = match.group(3)
        partition = match.group(4)
        
        # Convert gpu_blocks to int if numeric
        if gpu_blocks.isdigit():
            gpu_blocks = int(gpu_blocks)
        
        # Convert partition to float if numeric
        if partition != 'dynamic':
            try:
                partition = float(partition)
            except ValueError:
                pass
        
        return {
            'benchmark': bench,
            'threads': threads,
            'gpu_blocks': gpu_blocks,
            'partition': partition
        }
    
    return None


def process_results_dir(results_dir):
    """
    Process all nsys-rep files in a results directory and create summary CSVs.
    """
    results_dir = Path(results_dir)
    
    # Find all nsys-rep files
    nsys_files = list(results_dir.glob('*.nsys-rep'))
    
    if not nsys_files:
        print(f"No .nsys-rep files found in {results_dir}")
        return
    
    print(f"Found {len(nsys_files)} nsys-rep files")
    
    # Group by benchmark
    benchmarks = {}
    
    for nsys_file in nsys_files:
        config = parse_filename(str(nsys_file))
        if config is None:
            print(f"  Skipping unrecognized file: {nsys_file.name}")
            continue
        
        bench = config['benchmark']
        if bench not in benchmarks:
            benchmarks[bench] = []
        
        print(f"  Parsing {nsys_file.name}...")
        
        # Try nsys stats first, fall back to SQLite
        pf_data = parse_nsys_report(str(nsys_file))
        if pf_data is None:
            pf_data = parse_sqlite_report(str(nsys_file))
        
        if pf_data is None:
            pf_data = {
                'cpu_page_faults': 'NA',
                'gpu_page_faults': 'NA',
                'cpu_pf_data_mb': 'NA',
                'gpu_pf_data_mb': 'NA',
            }
        
        benchmarks[bench].append({
            **config,
            **pf_data
        })
    
    # Write summary CSVs for each benchmark
    for bench, records in benchmarks.items():
        csv_file = results_dir / f'{bench}_pagefaults.csv'
        
        # Sort records by threads, gpu_blocks, partition
        def sort_key(r):
            gpu = r['gpu_blocks'] if isinstance(r['gpu_blocks'], int) else 0
            part = r['partition'] if isinstance(r['partition'], float) else -1
            return (r['threads'], gpu, part)
        
        records.sort(key=sort_key)
        
        # Write CSV
        fieldnames = ['benchmark', 'threads', 'gpu_blocks', 'partition',
                      'cpu_page_faults', 'gpu_page_faults', 
                      'htod_migration_mb', 'dtoh_migration_mb',
                      'cpu_pf_data_mb', 'gpu_pf_data_mb']
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
        
        print(f"  Wrote {csv_file}")
    
    print(f"\nSummary CSVs written to {results_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_pagefault_results.py <results_dir>")
        print("")
        print("Processes nsys-rep files and creates summary CSV files with page fault counts.")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)
    
    process_results_dir(results_dir)


if __name__ == '__main__':
    main()
