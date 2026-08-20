# Helix Performance Notebook

Helix publishes only measurements produced by executable benchmarks. This document
tracks the methodology and the conclusions from those measurements; it does not use
illustrative or estimated values in result columns.

## Required environment record

Every recorded experiment must include:

- date and Helix commit,
- CPU, RAM, and operating system,
- GPU and driver when applicable,
- CUDA toolkit and library versions when applicable,
- compiler and CMake versions,
- build type and relevant CMake options,
- dtype, tensor/model dimensions, batch size, and sequence length,
- warmup iterations, measured iterations, and repetitions,
- timing mechanism and synchronization boundary.

## Experiment log

| Experiment | Baseline | Change | Result | Interpretation |
|---|---|---|---|---|
| M1 `Tensor::zeros` baseline | No earlier implementation | Allocate 64-byte-aligned CPU storage and zero it | Median CPU time: 191 ns (1 float), 240 ns (1,024 floats), 23.589 us (1,048,576 floats) | This measures allocation, tensor metadata construction, and zeroing together. It is a starting point, not evidence of an optimization or a standalone memory-bandwidth result. |
| M2 CPU reference kernels | No earlier implementation | Clear single-threaded Float32 correctness paths with preallocated output | Representative median CPU times: add 1,048,576 values 121.685 us; softmax 32x128 21.172 us; matmul 128x128x128 1.521 ms (2.758 GFLOP/s); RMSNorm 32x4096 132.403 us | These establish correctness-oracle baselines. They are not optimized CPU-library comparisons, and allocation is excluded through `_out` APIs. |

## M1 CPU tensor baseline — 2026-08-20

Environment:

- source: commit `2887901`
- host: Apple M3 Pro, 12 CPU cores, 18 GB RAM
- OS: macOS 26.0.1 arm64
- compiler: Apple Clang 17.0.0
- CMake: 4.2.3
- Google Benchmark: 1.9.1
- build: `Release`, `HELIX_ENABLE_CUDA=OFF`, Float32
- workload: construct `Tensor::zeros({N})`; allocation and destruction are included
- warmup: time-based 0.1 seconds per benchmark case
- measurement: time-based minimum 0.2 seconds per repetition, five repetitions;
  median CPU time reported
- concurrency: one benchmark thread

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  --benchmark_filter=BM_TensorZeros \
  --benchmark_min_warmup_time=0.1 \
  --benchmark_min_time=0.2s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

The host load average was high (14.05 one-minute) and Google Benchmark could not
read Apple Silicon clock-rate metadata or set affinity. These results are retained
as an honest local baseline and should not be used for cross-machine comparisons.
The apparent bytes/second counter is intentionally omitted because allocation,
page state, cache reuse, and zeroing are combined in this fixture.

## M2 CPU reference-kernel baseline — 2026-08-20

Environment:

- source: commit `1488acb`
- host: Apple M3 Pro, 12 CPU cores, 18 GB RAM
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, Float32
- Google Benchmark: 1.9.1
- output allocation: excluded; every fixture reuses a preallocated output tensor
- warmup: time-based 0.1 seconds per case
- measurement: time-based minimum 0.2 seconds per repetition, five repetitions;
  median CPU time reported
- concurrency: one benchmark thread; kernels themselves are single-threaded

Selected results:

| Operation and shape | Median CPU time | Derived rate |
|---|---:|---:|
| Add, 1,024 values | 206 ns | 4.971 billion values/s |
| Add, 1,048,576 values | 121.685 us | 8.617 billion values/s |
| SiLU, 1,024 values | 3.081 us | 332.4 million values/s |
| SiLU, 1,048,576 values | 6.899 ms | 152.0 million values/s |
| Softmax, 32 x 128 | 21.172 us | 193.5 million values/s |
| Softmax, 16 x 1,023 (odd width) | 86.639 us | 188.9 million values/s |
| Matmul, 31 x 29 times 29 x 37 | 13.643 us | 4.876 GFLOP/s |
| Matmul, 64 x 64 times 64 x 64 | 151.980 us | 3.450 GFLOP/s |
| Matmul, 128 x 128 times 128 x 128 | 1.521 ms | 2.758 GFLOP/s |
| RMSNorm, 1 x 4,096 | 4.309 us | 950.7 million values/s |
| RMSNorm, 32 x 4,096 | 132.403 us | 990.0 million values/s |

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  --benchmark_filter=BM_Cpu \
  --benchmark_min_warmup_time=0.1 \
  --benchmark_min_time=0.2s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

The one-minute load average was 11.31. As in M1, clock-rate metadata and affinity
were unavailable on this host, so the measurements are local baselines rather than
cross-machine claims. The declining naive-matmul rate as size grows is expected
from its unblocked `i,j,k` loop and strided access down columns of the right-hand
matrix. That observation motivates the separate tiled CUDA progression; the CPU
oracle is kept simple and numerically conservative with double accumulation.

## Timing rules

CPU microbenchmarks use Google Benchmark and a release build. Inputs are kept alive
with benchmark barriers where required, and setup/allocation is separated unless it
is the operation under study.

CUDA kernel timing uses warmed-up kernels and CUDA events recorded on the same
stream as the measured work. End-to-end host latency, transfer latency, and kernel
execution time are reported separately. A kernel launch measured without the
required synchronization is never reported as execution latency.

## Local benchmark commands

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DHELIX_ENABLE_CUDA=OFF
cmake --build build-release -j
./build-release/benchmarks/helix_benchmarks
```

The dtype-metadata benchmark is a harness smoke test, not a runtime performance
claim.
