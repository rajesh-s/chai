# CUDA-U Benchmarks

Heterogeneous CPU-GPU benchmarks from the Chai benchmark suite.

## Benchmarks

| Benchmark | Description |
|-----------|-------------|
| BFS | Breadth-First Search |
| BS | Bézier Surface |
| CEDD | Canny Edge Detection (Data partitioning) |
| CEDT | Canny Edge Detection (Task partitioning) |
| HSTI | Histogram (Input partitioning) |
| HSTO | Histogram (Output partitioning) |
| PAD | Padding |
| RSCD | RANSAC (Data partitioning) |
| RSCT | RANSAC (Task partitioning) |
| SC | Stream Compaction |
| SSSP | Single-Source Shortest Path |
| TQ | Task Queue |
| TQH | Task Queue Histogram |
| TRNS | Transpose |

## Building

### Build all benchmarks

```bash
./build_all_benchmarks.sh [sm_90|sm_90a]
```

- Default architecture: `sm_90a`
- Use `sm_90` for standard Hopper GPUs
- Use `sm_90a` for GH200 with hardware coherence features

### Build individual benchmark

```bash
cd <BENCHMARK>
make clean && make
```

## Running Coherence Study

Run all benchmarks with varying CPU threads and partition fractions:

```bash
./run_coherence_study.sh
```

Results are saved to a timestamped directory. Plot results with:

```bash
python3 plot_coherence_results.py <results_dir> "Platform Name"
```

## Requirements

- CUDA Toolkit (nvcc)
- Set `CHAI_CUDA_LIB` environment variable to CUDA library path
