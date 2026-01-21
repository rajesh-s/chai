#!/bin/bash
# Test script to verify each benchmark works with one configuration
# Run this BEFORE the full coherence study

echo "=== Testing All Benchmarks ==="
echo "Testing with: threads=4, gpu_blocks=8, partition=0.5, warmup=1, reps=1"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PASS=0
FAIL=0

test_benchmark() {
    local name=$1
    local dir=$2
    local exe=$3
    local args=$4
    
    echo -n "Testing $name... "
    
    # Run from benchmark's directory for correct input paths
    output=$(cd "$SCRIPT_DIR/$dir" && timeout 600 ./$exe $args 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "TIMEOUT (10m)"
        FAIL=$((FAIL + 1))
        return 1
    elif echo "$output" | grep -q "Kernel Time"; then
        kernel_time=$(echo "$output" | grep "Kernel Time" | grep -oE "[0-9]+\.[0-9]+")
        echo "OK (Kernel Time: ${kernel_time}ms)"
        PASS=$((PASS + 1))
        return 0
    elif echo "$output" | grep -qi "error"; then
        error_msg=$(echo "$output" | grep -i "error" | head -1)
        echo "ERROR: $error_msg"
        FAIL=$((FAIL + 1))
        return 1
    else
        echo "UNKNOWN OUTPUT:"
        echo "$output" | head -5
        FAIL=$((FAIL + 1))
        return 1
    fi
}

# Test each benchmark with minimal config using default datasets
echo "--- Benchmarks with partition (-a) + GPU blocks (-g) ---"
test_benchmark "BS"   "BS"   "bs"   "-f input/control.txt -t 4 -g 8 -a 0.5 -w 1 -r 1"
test_benchmark "HSTI" "HSTI" "hsti" "-t 4 -g 8 -a 0.5 -w 1 -r 1"
test_benchmark "HSTO" "HSTO" "hsto" "-t 4 -g 8 -a 0.5 -w 1 -r 1"
test_benchmark "PAD"  "PAD"  "pad"  "-t 4 -g 8 -a 0.5 -w 1 -r 1"
test_benchmark "RSCD" "RSCD" "rscd" "-f input/vectors.csv -t 4 -g 8 -a 0.5 -w 1 -r 1"
test_benchmark "SC"   "SC"   "sc"   "-t 4 -g 8 -a 0.5 -w 1 -r 1"

echo ""
echo "--- Benchmarks with partition (-a) but NO GPU blocks ---"
test_benchmark "CEDD" "CEDD" "cedd" "-f input/peppa/ -t 4 -a 0.5 -w 1 -r 1"

echo ""
echo "--- Benchmarks with GPU blocks (-g) but NO partition (dynamic/task-based) ---"
test_benchmark "BFS"  "BFS"  "bfs"  "-f input/NYR_input.dat -l 128 -t 4 -g 8 -w 1 -r 1"
test_benchmark "SSSP" "SSSP" "sssp" "-f input/NYR_input.dat -l 128 -t 4 -g 8 -w 1 -r 1"
test_benchmark "RSCT" "RSCT" "rsct" "-f input/vectors.csv -t 4 -g 8 -w 1 -r 1"
test_benchmark "TQ"   "TQ"   "tq"   "-f input/patternsNP100NB512FB50.txt -t 4 -g 8 -w 1 -r 1"
test_benchmark "TQH"  "TQH"  "tqh"  "-f input/basket/basket -t 4 -g 8 -w 1 -r 1"
test_benchmark "TRNS" "TRNS" "trns" "-t 4 -g 8 -w 1 -r 1"

echo ""
echo "--- Benchmarks with threads only (no GPU blocks support) ---"
test_benchmark "CEDT" "CEDT" "cedt" "-f input/peppa/ -t 4 -w 1 -r 1"

echo ""
echo "=== Summary ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "All benchmarks passed! Safe to run full coherence study."
    exit 0
else
    echo "Some benchmarks failed. Please fix before running full study."
    exit 1
fi
