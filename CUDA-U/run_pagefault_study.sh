#!/bin/bash
# Page Fault Study: HW (GH200) vs SW (x86+H100)
# Uses nsys to capture CUDA Unified Memory page faults
# Varies: CPU threads (-t) and partition fraction (-a)
#
# Usage: ./run_pagefault_study.sh <system_type> [output_dir]
#        system_type: gh200 or h100 (MANDATORY)

# Get script directory first
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check mandatory system type argument
if [ -z "$1" ]; then
    echo "Error: System type is mandatory"
    echo "Usage: ./run_pagefault_study.sh <system_type> [output_dir]"
    echo "  system_type: gh200 or h100"
    exit 1
fi

SYSTEM_TYPE="$1"

# Validate system type
if [ "$SYSTEM_TYPE" != "gh200" ] && [ "$SYSTEM_TYPE" != "h100" ]; then
    echo "Error: Invalid system type '$SYSTEM_TYPE'"
    echo "Must be either 'gh200' or 'h100'"
    exit 1
fi

# Output directory based on second argument (optional)
if [ -z "$2" ]; then
    OUTDIR="$SCRIPT_DIR/pagefault_results_${SYSTEM_TYPE}_$(hostname)_$(date +%Y%m%d_%H%M%S)"
else
    OUTDIR="$SCRIPT_DIR/$2"
fi

mkdir -p "$OUTDIR"
echo "$SYSTEM_TYPE" > "$OUTDIR/system_type.txt"

# Configuration
THREADS=(1 4 16 32 64)
PARTITIONS=(0.0 0.1 0.25 0.5 0.75 0.9 1.0)
GPU_BLOCKS=(8 16 32 64)
WARMUP=1
REPS=3  # Fewer reps for profiling (slower)
TIMEOUT=900  # 15 minutes timeout (profiling takes longer)

# Check for nsys
if ! command -v nsys &> /dev/null; then
    echo "Error: nsys not found. Install with: sudo apt install nsight-systems"
    exit 1
fi

# ============================================================================
# Benchmark configurations (same as run_coherence_study.sh)
# ============================================================================

BFS_ARGS="-f input/NYR_input.dat -l 128"
BS_ARGS="-f input/control.txt"
CEDD_ARGS="-f input/peppa/"
CEDT_ARGS="-f input/peppa/"
HSTI_ARGS=""
HSTO_ARGS=""
PAD_ARGS=""
RSCD_ARGS="-f input/vectors.csv"
RSCT_ARGS="-f input/vectors.csv"
SC_ARGS=""
SSSP_ARGS="-f input/NYR_input.dat -l 128"
TQ_ARGS="-f input/patternsNP100NB512FB50.txt"
TQH_ARGS="-f input/basket/basket"
TRNS_ARGS=""

# ============================================================================
# Helper functions
# ============================================================================

run_with_nsys() {
    local bench=$1
    local bench_dir=$2
    local exe=$3
    local extra_args=$4
    local threads=$5
    local gpu_blocks=$6
    local partition=$7
    local report_prefix=$8
    
    local nsys_output="${OUTDIR}/${bench}_t${threads}_g${gpu_blocks}_a${partition}"
    
    # Build command based on available options
    local cmd="./$exe -t $threads -w $WARMUP -r $REPS"
    
    if [ "$gpu_blocks" != "default" ]; then
        cmd="$cmd -g $gpu_blocks"
    fi
    
    if [ "$partition" != "dynamic" ]; then
        cmd="$cmd -a $partition"
    fi
    
    cmd="$cmd $extra_args"
    
    # Run with nsys - capture CUDA UM events
    # --cuda-um-cpu-page-faults=true captures CPU page faults for UM
    # --cuda-um-gpu-page-faults=true captures GPU page faults for UM
    cd "$SCRIPT_DIR/$bench_dir" && timeout $TIMEOUT nsys profile \
        --output="${nsys_output}" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --cuda-um-cpu-page-faults=true \
        --cuda-um-gpu-page-faults=true \
        $cmd > "${nsys_output}_stdout.txt" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "TIMEOUT" > "${nsys_output}_status.txt"
        return 1
    elif [ $exit_code -ne 0 ]; then
        echo "ERROR:$exit_code" > "${nsys_output}_status.txt"
        return 1
    else
        echo "OK" > "${nsys_output}_status.txt"
    fi
    
    return 0
}

extract_pagefaults() {
    local nsys_file=$1
    local csv_output=$2
    
    # Export nsys report to SQLite and extract page fault stats
    if [ -f "${nsys_file}.nsys-rep" ]; then
        nsys stats --report cuda_um_cpu_page_faults --format csv "${nsys_file}.nsys-rep" > "${csv_output}_cpu_pf.csv" 2>/dev/null
        nsys stats --report cuda_um_gpu_page_faults --format csv "${nsys_file}.nsys-rep" > "${csv_output}_gpu_pf.csv" 2>/dev/null
        
        # Also export summary
        nsys stats --report cuda_gpu_kern_sum --format csv "${nsys_file}.nsys-rep" > "${csv_output}_kern.csv" 2>/dev/null
    fi
}

run_benchmark_with_partition() {
    local bench=$1
    local bench_dir=$2
    local exe=$3
    local extra_args=$4
    local csv_file=$5
    
    echo "=== Running $bench (with partition + GPU blocks sweep) ===" | tee -a "$OUTDIR/run.log"
    echo "benchmark,threads,gpu_blocks,partition,cpu_page_faults,gpu_page_faults,cpu_pf_data_mb,gpu_pf_data_mb" > "$csv_file"
    
    for t in "${THREADS[@]}"; do
        for g in "${GPU_BLOCKS[@]}"; do
            for a in "${PARTITIONS[@]}"; do
                echo "  $bench: threads=$t, gpu_blocks=$g, partition=$a" | tee -a "$OUTDIR/run.log"
                
                run_with_nsys "$bench" "$bench_dir" "$exe" "$extra_args" "$t" "$g" "$a" ""
                
                nsys_file="${OUTDIR}/${bench}_t${t}_g${g}_a${a}"
                extract_pagefaults "$nsys_file" "$nsys_file"
            done
        done
    done
    echo "  Raw nsys reports saved to $OUTDIR" | tee -a "$OUTDIR/run.log"
}

run_benchmark_threads_only() {
    local bench=$1
    local bench_dir=$2
    local exe=$3
    local extra_args=$4
    local csv_file=$5
    
    echo "=== Running $bench (threads + GPU blocks sweep) ===" | tee -a "$OUTDIR/run.log"
    echo "benchmark,threads,gpu_blocks,partition,cpu_page_faults,gpu_page_faults,cpu_pf_data_mb,gpu_pf_data_mb" > "$csv_file"
    
    for t in "${THREADS[@]}"; do
        for g in "${GPU_BLOCKS[@]}"; do
            echo "  $bench: threads=$t, gpu_blocks=$g" | tee -a "$OUTDIR/run.log"
            
            run_with_nsys "$bench" "$bench_dir" "$exe" "$extra_args" "$t" "$g" "dynamic" ""
            
            nsys_file="${OUTDIR}/${bench}_t${t}_g${g}_adynamic"
            extract_pagefaults "$nsys_file" "$nsys_file"
        done
    done
    echo "  Raw nsys reports saved to $OUTDIR" | tee -a "$OUTDIR/run.log"
}

run_benchmark_partition_no_gpu() {
    local bench=$1
    local bench_dir=$2
    local exe=$3
    local extra_args=$4
    local csv_file=$5
    
    echo "=== Running $bench (partition sweep, no GPU blocks) ===" | tee -a "$OUTDIR/run.log"
    echo "benchmark,threads,gpu_blocks,partition,cpu_page_faults,gpu_page_faults,cpu_pf_data_mb,gpu_pf_data_mb" > "$csv_file"
    
    for t in "${THREADS[@]}"; do
        for a in "${PARTITIONS[@]}"; do
            echo "  $bench: threads=$t, partition=$a" | tee -a "$OUTDIR/run.log"
            
            run_with_nsys "$bench" "$bench_dir" "$exe" "$extra_args" "$t" "default" "$a" ""
            
            nsys_file="${OUTDIR}/${bench}_t${t}_gdefault_a${a}"
            extract_pagefaults "$nsys_file" "$nsys_file"
        done
    done
    echo "  Raw nsys reports saved to $OUTDIR" | tee -a "$OUTDIR/run.log"
}

run_benchmark_threads_no_gpu() {
    local bench=$1
    local bench_dir=$2
    local exe=$3
    local extra_args=$4
    local csv_file=$5
    
    echo "=== Running $bench (threads only, no GPU blocks) ===" | tee -a "$OUTDIR/run.log"
    echo "benchmark,threads,gpu_blocks,partition,cpu_page_faults,gpu_page_faults,cpu_pf_data_mb,gpu_pf_data_mb" > "$csv_file"
    
    for t in "${THREADS[@]}"; do
        echo "  $bench: threads=$t" | tee -a "$OUTDIR/run.log"
        
        run_with_nsys "$bench" "$bench_dir" "$exe" "$extra_args" "$t" "default" "dynamic" ""
        
        nsys_file="${OUTDIR}/${bench}_t${t}_gdefault_adynamic"
        extract_pagefaults "$nsys_file" "$nsys_file"
    done
    echo "  Raw nsys reports saved to $OUTDIR" | tee -a "$OUTDIR/run.log"
}

# ============================================================================
# Main execution
# ============================================================================

echo "=== Chai Page Fault Study (nsys) ===" | tee "$OUTDIR/run.log"
echo "Host: $(hostname)" | tee -a "$OUTDIR/run.log"
echo "Date: $(date)" | tee -a "$OUTDIR/run.log"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')" | tee -a "$OUTDIR/run.log"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'N/A')" | tee -a "$OUTDIR/run.log"
echo "nsys: $(nsys --version 2>/dev/null | head -1 || echo 'N/A')" | tee -a "$OUTDIR/run.log"
echo "Threads: ${THREADS[*]}" | tee -a "$OUTDIR/run.log"
echo "GPU Blocks: ${GPU_BLOCKS[*]}" | tee -a "$OUTDIR/run.log"
echo "Partitions: ${PARTITIONS[*]}" | tee -a "$OUTDIR/run.log"
echo "Warmup: $WARMUP, Reps: $REPS" | tee -a "$OUTDIR/run.log"
echo "" | tee -a "$OUTDIR/run.log"

# --- Benchmarks with partition sweep (-a flag) AND GPU blocks (-g) ---
run_benchmark_with_partition "BS" "BS" "bs" "$BS_ARGS" "$OUTDIR/BS_pagefaults.csv"
run_benchmark_with_partition "HSTI" "HSTI" "hsti" "$HSTI_ARGS" "$OUTDIR/HSTI_pagefaults.csv"
run_benchmark_with_partition "HSTO" "HSTO" "hsto" "$HSTO_ARGS" "$OUTDIR/HSTO_pagefaults.csv"
run_benchmark_with_partition "PAD" "PAD" "pad" "$PAD_ARGS" "$OUTDIR/PAD_pagefaults.csv"
run_benchmark_with_partition "RSCD" "RSCD" "rscd" "$RSCD_ARGS" "$OUTDIR/RSCD_pagefaults.csv"
run_benchmark_with_partition "SC" "SC" "sc" "$SC_ARGS" "$OUTDIR/SC_pagefaults.csv"

# --- Benchmarks with partition (-a) but NO GPU blocks support ---
run_benchmark_partition_no_gpu "CEDD" "CEDD" "cedd" "$CEDD_ARGS" "$OUTDIR/CEDD_pagefaults.csv"

# --- Benchmarks with threads + GPU blocks (task-based / dynamic / graph traversal) ---
run_benchmark_threads_only "BFS" "BFS" "bfs" "$BFS_ARGS" "$OUTDIR/BFS_pagefaults.csv"
run_benchmark_threads_only "SSSP" "SSSP" "sssp" "$SSSP_ARGS" "$OUTDIR/SSSP_pagefaults.csv"
run_benchmark_threads_only "RSCT" "RSCT" "rsct" "$RSCT_ARGS" "$OUTDIR/RSCT_pagefaults.csv"
run_benchmark_threads_only "TQ" "TQ" "tq" "$TQ_ARGS" "$OUTDIR/TQ_pagefaults.csv"
run_benchmark_threads_only "TQH" "TQH" "tqh" "$TQH_ARGS" "$OUTDIR/TQH_pagefaults.csv"
run_benchmark_threads_only "TRNS" "TRNS" "trns" "$TRNS_ARGS" "$OUTDIR/TRNS_pagefaults.csv"

# --- Benchmarks with threads only (no GPU blocks support) ---
run_benchmark_threads_no_gpu "CEDT" "CEDT" "cedt" "$CEDT_ARGS" "$OUTDIR/CEDT_pagefaults.csv"

# ============================================================================
# Post-processing: Parse nsys reports and create summary CSVs
# ============================================================================

echo "" | tee -a "$OUTDIR/run.log"
echo "=== Post-processing nsys reports ===" | tee -a "$OUTDIR/run.log"
python3 "$SCRIPT_DIR/parse_pagefault_results.py" "$OUTDIR" "$SYSTEM_TYPE"

# ============================================================================
# Summary
# ============================================================================

echo "" | tee -a "$OUTDIR/run.log"
echo "=== Complete ===" | tee -a "$OUTDIR/run.log"
echo "Results directory: $OUTDIR" | tee -a "$OUTDIR/run.log"
echo "" | tee -a "$OUTDIR/run.log"
echo "To plot results:" | tee -a "$OUTDIR/run.log"
echo "  python3 plot_pagefault_results.py $OUTDIR 'GH200 (HW Coherence)'" | tee -a "$OUTDIR/run.log"
echo "" | tee -a "$OUTDIR/run.log"
echo "To compare two systems:" | tee -a "$OUTDIR/run.log"
echo "  python3 plot_pagefault_results.py <gh200_dir> <x86_h100_dir> 'GH200' 'x86+H100'" | tee -a "$OUTDIR/run.log"
