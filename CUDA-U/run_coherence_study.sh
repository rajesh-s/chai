#!/bin/bash
# Coherence Study: HW (GH200) vs SW (x86+H100)
# Varies: CPU threads (-t) and partition fraction (-a)
# Parameters based on ISPASS'17 paper Table 3

# Output directory
OUTDIR="results_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

# Configuration
THREADS=(1 4 16 32 64)
PARTITIONS=(0.0 0.1 0.25 0.5 0.75 0.9 1.0)
GPU_BLOCKS=(8 16 32 64)  # Sweep GPU blocks (limited to 64 for persistent thread safety)
WARMUP=5
REPS=20
TIMEOUT=600  # 10 minutes timeout per benchmark run

# ============================================================================
# Benchmark configurations from ISPASS'17 Paper Table 3
# ============================================================================
# Benchmark configurations - using default datasets
# ============================================================================

# BFS - Breadth-First Search
# Input: USA-road-d.NY graph (264K nodes, 730K edges)
# Note: Uses persistent threads, GPU blocks swept separately
BFS_ARGS="-f input/NYR_input.dat -l 128"

# BS - Bézier Surface  
# Input: 4x4 control points from file, 300x300 output resolution (default)
BS_ARGS="-f input/control.txt"

# CEDD - Canny Edge Detection (Data partitioning)
# Input: Video frames (peppa - default)
CEDD_ARGS="-f input/peppa/"

# CEDT - Canny Edge Detection (Task partitioning)
# Input: Video frames (peppa - default)
CEDT_ARGS="-f input/peppa/"

# HSTI - Histogram (Input partitioning)
# Input: default 1,572,864 pixels, 256 bins
HSTI_ARGS=""

# HSTO - Histogram (Output partitioning)
# Input: default 1,572,864 pixels, 256 bins
HSTO_ARGS=""

# PAD - Padding
# Input: default 1000x999 matrix
PAD_ARGS=""

# RSCD - RANSAC (Data partitioning)
# Input: default vectors.csv
RSCD_ARGS="-f input/vectors.csv"

# RSCT - RANSAC (Task partitioning)
# Input: default vectors.csv
RSCT_ARGS="-f input/vectors.csv"

# SC - Stream Compaction
# Input: default 65536 elements, 50% compaction
SC_ARGS=""

# SSSP - Single-Source Shortest Path
# Input: USA-road-d.NY graph (264K nodes, 730K edges)
# Note: Uses persistent threads, GPU blocks swept separately
SSSP_ARGS="-f input/NYR_input.dat -l 128"

# TQ - Task Queue
# Input: default patterns file (FB50 = 50% feedback)
TQ_ARGS="-f input/patternsNP100NB512FB50.txt"

# TQH - Task Queue Histogram
# Input: default basket video
TQH_ARGS="-f input/basket/basket"

# TRNS - Transpose
# Input: default 8000x8000 matrix
TRNS_ARGS=""

# ============================================================================
# Helper functions
# ============================================================================

run_benchmark_with_partition() {
    local bench=$1
    local bench_dir=$2
    local exe=$3
    local extra_args=$4
    local csv_file=$5
    
    echo "=== Running $bench (with partition + GPU blocks sweep) ===" | tee -a "$OUTDIR/run.log"
    echo "benchmark,threads,gpu_blocks,partition,kernel_time_ms,alloc_time_ms,init_time_ms,dealloc_time_ms" > "$csv_file"
    
    for t in "${THREADS[@]}"; do
        for g in "${GPU_BLOCKS[@]}"; do
            for a in "${PARTITIONS[@]}"; do
                echo "  $bench: threads=$t, gpu_blocks=$g, partition=$a" | tee -a "$OUTDIR/run.log"
                
                output=$(cd "$SCRIPT_DIR/$bench_dir" && timeout $TIMEOUT ./$exe -t $t -g $g -a $a -w $WARMUP -r $REPS $extra_args 2>&1)
                exit_code=$?
                
                if [ $exit_code -eq 124 ]; then
                    echo "    TIMEOUT after ${TIMEOUT}s" | tee -a "$OUTDIR/run.log"
                    kernel_time="TIMEOUT"
                    alloc_time="TIMEOUT"
                    init_time="TIMEOUT"
                    dealloc_time="TIMEOUT"
                else
                    kernel_time=$(echo "$output" | grep "Kernel Time" | grep -oE "[0-9]+\.[0-9]+")
                    alloc_time=$(echo "$output" | grep "Allocation Time" | grep -oE "[0-9]+\.[0-9]+")
                    init_time=$(echo "$output" | grep "Initialization Time" | grep -oE "[0-9]+\.[0-9]+")
                    dealloc_time=$(echo "$output" | grep "Deallocation Time" | grep -oE "[0-9]+\.[0-9]+")
                    
                    [ -z "$kernel_time" ] && kernel_time="NA"
                    [ -z "$alloc_time" ] && alloc_time="NA"
                    [ -z "$init_time" ] && init_time="NA"
                    [ -z "$dealloc_time" ] && dealloc_time="NA"
                fi
                
                echo "$bench,$t,$g,$a,$kernel_time,$alloc_time,$init_time,$dealloc_time" >> "$csv_file"
                
                echo "--- threads=$t gpu_blocks=$g partition=$a ---" >> "$OUTDIR/${bench}_raw.log"
                echo "$output" >> "$OUTDIR/${bench}_raw.log"
            done
        done
    done
    echo "  Results saved to $csv_file" | tee -a "$OUTDIR/run.log"
}

run_benchmark_threads_only() {
    local bench=$1
    local bench_dir=$2
    local exe=$3
    local extra_args=$4
    local csv_file=$5
    
    echo "=== Running $bench (threads + GPU blocks sweep) ===" | tee -a "$OUTDIR/run.log"
    echo "benchmark,threads,gpu_blocks,partition,kernel_time_ms,alloc_time_ms,init_time_ms,dealloc_time_ms" > "$csv_file"
    
    for t in "${THREADS[@]}"; do
        for g in "${GPU_BLOCKS[@]}"; do
            echo "  $bench: threads=$t, gpu_blocks=$g" | tee -a "$OUTDIR/run.log"
            
            output=$(cd "$SCRIPT_DIR/$bench_dir" && timeout $TIMEOUT ./$exe -t $t -g $g -w $WARMUP -r $REPS $extra_args 2>&1)
            exit_code=$?
            
            if [ $exit_code -eq 124 ]; then
                echo "    TIMEOUT after ${TIMEOUT}s" | tee -a "$OUTDIR/run.log"
                kernel_time="TIMEOUT"
                alloc_time="TIMEOUT"
                init_time="TIMEOUT"
                dealloc_time="TIMEOUT"
            else
                kernel_time=$(echo "$output" | grep "Kernel Time" | grep -oE "[0-9]+\.[0-9]+")
                alloc_time=$(echo "$output" | grep "Allocation Time" | grep -oE "[0-9]+\.[0-9]+")
                init_time=$(echo "$output" | grep "Initialization Time" | grep -oE "[0-9]+\.[0-9]+")
                dealloc_time=$(echo "$output" | grep "Deallocation Time" | grep -oE "[0-9]+\.[0-9]+")
                
                [ -z "$kernel_time" ] && kernel_time="NA"
                [ -z "$alloc_time" ] && alloc_time="NA"
                [ -z "$init_time" ] && init_time="NA"
                [ -z "$dealloc_time" ] && dealloc_time="NA"
            fi
            
            echo "$bench,$t,$g,dynamic,$kernel_time,$alloc_time,$init_time,$dealloc_time" >> "$csv_file"
            
            echo "--- threads=$t gpu_blocks=$g ---" >> "$OUTDIR/${bench}_raw.log"
            echo "$output" >> "$OUTDIR/${bench}_raw.log"
        done
    done
    echo "  Results saved to $csv_file" | tee -a "$OUTDIR/run.log"
}

# For benchmarks that support partition (-a) but NOT GPU blocks (-g)
run_benchmark_partition_no_gpu() {
    local bench=$1
    local bench_dir=$2
    local exe=$3
    local extra_args=$4
    local csv_file=$5
    
    echo "=== Running $bench (partition sweep, no GPU blocks) ===" | tee -a "$OUTDIR/run.log"
    echo "benchmark,threads,gpu_blocks,partition,kernel_time_ms,alloc_time_ms,init_time_ms,dealloc_time_ms" > "$csv_file"
    
    for t in "${THREADS[@]}"; do
        for a in "${PARTITIONS[@]}"; do
            echo "  $bench: threads=$t, partition=$a" | tee -a "$OUTDIR/run.log"
            
            output=$(cd "$SCRIPT_DIR/$bench_dir" && timeout $TIMEOUT ./$exe -t $t -a $a -w $WARMUP -r $REPS $extra_args 2>&1)
            exit_code=$?
            
            if [ $exit_code -eq 124 ]; then
                echo "    TIMEOUT after ${TIMEOUT}s" | tee -a "$OUTDIR/run.log"
                kernel_time="TIMEOUT"
                alloc_time="TIMEOUT"
                init_time="TIMEOUT"
                dealloc_time="TIMEOUT"
            else
                kernel_time=$(echo "$output" | grep "Kernel Time" | grep -oE "[0-9]+\.[0-9]+")
                alloc_time=$(echo "$output" | grep "Allocation Time" | grep -oE "[0-9]+\.[0-9]+")
                init_time=$(echo "$output" | grep "Initialization Time" | grep -oE "[0-9]+\.[0-9]+")
                dealloc_time=$(echo "$output" | grep "Deallocation Time" | grep -oE "[0-9]+\.[0-9]+")
                
                [ -z "$kernel_time" ] && kernel_time="NA"
                [ -z "$alloc_time" ] && alloc_time="NA"
                [ -z "$init_time" ] && init_time="NA"
                [ -z "$dealloc_time" ] && dealloc_time="NA"
            fi
            
            echo "$bench,$t,default,$a,$kernel_time,$alloc_time,$init_time,$dealloc_time" >> "$csv_file"
            
            echo "--- threads=$t partition=$a ---" >> "$OUTDIR/${bench}_raw.log"
            echo "$output" >> "$OUTDIR/${bench}_raw.log"
        done
    done
    echo "  Results saved to $csv_file" | tee -a "$OUTDIR/run.log"
}

# For benchmarks that DON'T support partition or GPU blocks (threads only)
run_benchmark_threads_no_gpu() {
    local bench=$1
    local bench_dir=$2
    local exe=$3
    local extra_args=$4
    local csv_file=$5
    
    echo "=== Running $bench (threads only, no GPU blocks) ===" | tee -a "$OUTDIR/run.log"
    echo "benchmark,threads,gpu_blocks,partition,kernel_time_ms,alloc_time_ms,init_time_ms,dealloc_time_ms" > "$csv_file"
    
    for t in "${THREADS[@]}"; do
        echo "  $bench: threads=$t" | tee -a "$OUTDIR/run.log"
        
        output=$(cd "$SCRIPT_DIR/$bench_dir" && timeout $TIMEOUT ./$exe -t $t -w $WARMUP -r $REPS $extra_args 2>&1)
        exit_code=$?
        
        if [ $exit_code -eq 124 ]; then
            echo "    TIMEOUT after ${TIMEOUT}s" | tee -a "$OUTDIR/run.log"
            kernel_time="TIMEOUT"
            alloc_time="TIMEOUT"
            init_time="TIMEOUT"
            dealloc_time="TIMEOUT"
        else
            kernel_time=$(echo "$output" | grep "Kernel Time" | grep -oE "[0-9]+\.[0-9]+")
            alloc_time=$(echo "$output" | grep "Allocation Time" | grep -oE "[0-9]+\.[0-9]+")
            init_time=$(echo "$output" | grep "Initialization Time" | grep -oE "[0-9]+\.[0-9]+")
            dealloc_time=$(echo "$output" | grep "Deallocation Time" | grep -oE "[0-9]+\.[0-9]+")
            
            [ -z "$kernel_time" ] && kernel_time="NA"
            [ -z "$alloc_time" ] && alloc_time="NA"
            [ -z "$init_time" ] && init_time="NA"
            [ -z "$dealloc_time" ] && dealloc_time="NA"
        fi
        
        echo "$bench,$t,default,dynamic,$kernel_time,$alloc_time,$init_time,$dealloc_time" >> "$csv_file"
        
        echo "--- threads=$t ---" >> "$OUTDIR/${bench}_raw.log"
        echo "$output" >> "$OUTDIR/${bench}_raw.log"
    done
    echo "  Results saved to $csv_file" | tee -a "$OUTDIR/run.log"
}

# ============================================================================
# Main execution
# ============================================================================

echo "=== Chai Coherence Study ===" | tee "$OUTDIR/run.log"
echo "Host: $(hostname)" | tee -a "$OUTDIR/run.log"
echo "Date: $(date)" | tee -a "$OUTDIR/run.log"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "$OUTDIR/run.log"
echo "CUDA: $(nvcc --version | grep release)" | tee -a "$OUTDIR/run.log"
echo "Threads: ${THREADS[*]}" | tee -a "$OUTDIR/run.log"
echo "GPU Blocks: ${GPU_BLOCKS[*]}" | tee -a "$OUTDIR/run.log"
echo "Partitions: ${PARTITIONS[*]}" | tee -a "$OUTDIR/run.log"
echo "Warmup: $WARMUP, Reps: $REPS" | tee -a "$OUTDIR/run.log"
echo "" | tee -a "$OUTDIR/run.log"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Benchmarks with partition sweep (-a flag) AND GPU blocks (-g) ---
run_benchmark_with_partition "BS" "BS" "bs" "$BS_ARGS" "$OUTDIR/BS_results.csv"
run_benchmark_with_partition "HSTI" "HSTI" "hsti" "$HSTI_ARGS" "$OUTDIR/HSTI_results.csv"
run_benchmark_with_partition "HSTO" "HSTO" "hsto" "$HSTO_ARGS" "$OUTDIR/HSTO_results.csv"
run_benchmark_with_partition "PAD" "PAD" "pad" "$PAD_ARGS" "$OUTDIR/PAD_results.csv"
run_benchmark_with_partition "RSCD" "RSCD" "rscd" "$RSCD_ARGS" "$OUTDIR/RSCD_results.csv"
run_benchmark_with_partition "SC" "SC" "sc" "$SC_ARGS" "$OUTDIR/SC_results.csv"

# --- Benchmarks with partition (-a) but NO GPU blocks support ---
run_benchmark_partition_no_gpu "CEDD" "CEDD" "cedd" "$CEDD_ARGS" "$OUTDIR/CEDD_results.csv"

# --- Benchmarks with threads + GPU blocks (task-based / dynamic / graph traversal) ---
run_benchmark_threads_only "BFS" "BFS" "bfs" "$BFS_ARGS" "$OUTDIR/BFS_results.csv"
run_benchmark_threads_only "SSSP" "SSSP" "sssp" "$SSSP_ARGS" "$OUTDIR/SSSP_results.csv"
run_benchmark_threads_only "RSCT" "RSCT" "rsct" "$RSCT_ARGS" "$OUTDIR/RSCT_results.csv"
run_benchmark_threads_only "TQ" "TQ" "tq" "$TQ_ARGS" "$OUTDIR/TQ_results.csv"
run_benchmark_threads_only "TQH" "TQH" "tqh" "$TQH_ARGS" "$OUTDIR/TQH_results.csv"
run_benchmark_threads_only "TRNS" "TRNS" "trns" "$TRNS_ARGS" "$OUTDIR/TRNS_results.csv"

# --- Benchmarks with threads only (no GPU blocks support) ---
run_benchmark_threads_no_gpu "CEDT" "CEDT" "cedt" "$CEDT_ARGS" "$OUTDIR/CEDT_results.csv"

# ============================================================================
# Summary
# ============================================================================

echo "" | tee -a "$OUTDIR/run.log"
echo "=== Complete ===" | tee -a "$OUTDIR/run.log"
echo "Results directory: $OUTDIR" | tee -a "$OUTDIR/run.log"
echo "" | tee -a "$OUTDIR/run.log"
echo "CSV files generated:" | tee -a "$OUTDIR/run.log"
ls -la "$OUTDIR"/*.csv | tee -a "$OUTDIR/run.log"
echo "" | tee -a "$OUTDIR/run.log"
echo "To plot results:" | tee -a "$OUTDIR/run.log"
echo "  python3 plot_coherence_results.py $OUTDIR 'GH200 (HW Coherence)'" | tee -a "$OUTDIR/run.log"
