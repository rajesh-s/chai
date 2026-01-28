#!/bin/bash
# Build all benchmarks in the CUDA-U directory
# Usage: ./build_all_benchmarks.sh [sm_90|sm_90a]

BENCHMARKS=(BFS BS CEDD CEDT HSTI HSTO PAD RSCD RSCT SC SSSP TQ TQH TRNS)

# Auto-detect CUDA library path
if [ -z "$CHAI_CUDA_LIB" ]; then
    if [ -d "/usr/lib/aarch64-linux-gnu" ] && [ -f "/usr/lib/aarch64-linux-gnu/libcudart.so" ]; then
        export CHAI_CUDA_LIB="/usr/lib/aarch64-linux-gnu"
    elif [ -d "/usr/local/cuda/lib64" ]; then
        export CHAI_CUDA_LIB="/usr/local/cuda/lib64"
    elif [ -d "/usr/lib/x86_64-linux-gnu" ] && [ -f "/usr/lib/x86_64-linux-gnu/libcudart.so" ]; then
        export CHAI_CUDA_LIB="/usr/lib/x86_64-linux-gnu"
    else
        echo "Error: Could not auto-detect CUDA library path. Set CHAI_CUDA_LIB manually."
        exit 1
    fi
    echo "Auto-detected CHAI_CUDA_LIB=$CHAI_CUDA_LIB"
fi

# Auto-detect CUDA include path
if [ -z "$CHAI_CUDA_INC" ]; then
    if [ -f "/usr/include/cuda_runtime.h" ]; then
        export CHAI_CUDA_INC="/usr/include"
    elif [ -d "/usr/local/cuda/include" ]; then
        export CHAI_CUDA_INC="/usr/local/cuda/include"
    else
        echo "Error: Could not auto-detect CUDA include path. Set CHAI_CUDA_INC manually."
        exit 1
    fi
    echo "Auto-detected CHAI_CUDA_INC=$CHAI_CUDA_INC"
fi

# GPU architecture (default: sm_90a)
ARCH="${1:-sm_90a}"

if [[ "$ARCH" != "sm_90" && "$ARCH" != "sm_90a" ]]; then
    echo "Usage: $0 [sm_90|sm_90a]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building all benchmarks with arch=$ARCH"

for bench in "${BENCHMARKS[@]}"; do
    if [ -f "$SCRIPT_DIR/$bench/Makefile" ]; then
        echo "Building $bench..."
        (cd "$SCRIPT_DIR/$bench" && make clean && make CXX_FLAGS="-std=c++17 -arch=$ARCH") || echo "FAILED: $bench"
    fi
done

echo "Done."
