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
| M7 host-resource allocator baseline | Direct aligned C++ allocation/free | Route identical fixed and 16-allocation mixed traces through the best-fit cache | Median CPU time per fixed pair: direct/Helix 22.6/87.1 ns (256 B), 22.0/88.6 ns (4 KiB), 548/88.2 ns (1 MiB); mixed trace 1.657/2.016 us | Host allocation already caches small blocks, so Helix metadata loses on small/mixed traces but avoids repeated upstream work for 1 MiB. This does not predict `cudaMalloc` behavior. |
| M9 CPU transformer primitives | No earlier RoPE fixture; M2 provided scalar SiLU only | Add preallocated SwiGLU and adjacent-pair RoPE reference paths | Median CPU time: SwiGLU 4,096 values 44.602 us and 131,072 values 1.512 ms; RoPE 1x32x128 1.904 us and 128x32x128 187.894 us | These are single-threaded correctness baselines, not optimized CPU or GPU comparisons. High host load makes wall-time variation substantial. |
| M10 contiguous CPU attention | No earlier attention path | Materialize scaled/masked scores, stable softmax in place, then probabilities x V with reusable output/workspace | Median CPU time: 8x1x128x64x64 76.371 us; 8x1x1,024x64x64 825.128 us; 3x17x29x37x41 46.284 us; 8x128x128x64x64 8.164 ms | Workspace is exactly `4*H*Q*K` bytes. These single-threaded results include softmax but the reported dot FLOP/s numerator does not; they are correctness baselines. |

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

## M7 host-resource allocator baseline — 2026-08-20

Environment:

- source: commit `746d0a5`
- host: Apple M3 Pro, 12 CPU cores, 18 GB RAM
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, warnings as errors
- Google Benchmark: 1.9.1
- repetitions: five; each repetition executes exactly 200,000 benchmark iterations
- warmup: time-based 0.05 seconds per case
- reported metric: median CPU time; one benchmark thread
- host load average at start: 13.30, 12.74, 12.46; affinity and CPU-frequency
  metadata were unavailable

Fixed-size results time one allocate/deallocate pair. `SystemMemoryResource` calls
aligned C++ `operator new/delete` every iteration; the Helix path retains a 64 MiB
default segment and uses its mutex-protected metadata cache after one upstream
allocation.

| Requested size | Direct resource | Helix cache | Observed outcome |
|---:|---:|---:|---|
| 256 B | 22.6 ns | 87.1 ns | Helix is 3.85x slower on this host path. |
| 4 KiB | 22.0 ns | 88.6 ns | Helix is 4.03x slower on this host path. |
| 1 MiB | 548 ns | 88.2 ns | Helix is 6.21x lower latency after cache warmup. |

The mixed trace allocates 16 blocks from 64 B through 256 KiB, then frees them in
reverse order. Direct allocation took 1.657 us per trace; Helix took 2.016 us and
retained 67,108,864 bytes with one upstream allocation. The cache was 21.7% slower
here, and the retained/requested ratio exposes that the GPU-oriented 64 MiB default
segment is excessive for this small host trace. These unfavorable results are
kept because they identify the mutex/vector/map bookkeeping cost and demonstrate
that caching is not intrinsically faster.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  --benchmark_filter='BM_(SystemResource|HelixAllocator)' \
  --benchmark_min_warmup_time=0.05 \
  --benchmark_min_time=200000x \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

The relevant production question is native `cudaMalloc/cudaFree` versus the CUDA-
backed cache under GPU allocation traces. Those fixtures compile, but the current
machine cannot execute them; no CUDA allocation latency or speedup is reported.

## M9 CPU transformer-primitive baseline — 2026-08-20

Environment:

- source: commit `bd5a1ab`
- host: Apple M3 Pro, 12 CPU cores, 18 GB RAM
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, Float32
- Google Benchmark: 1.9.1
- output allocation and cosine/sine construction: excluded from timed regions
- warmup: time-based 0.1 seconds per case
- measurement: time-based minimum 0.2 seconds per repetition, five repetitions;
  median CPU time reported
- one-minute load average at start: 29.32; affinity and CPU-frequency metadata were
  unavailable

| Operation and shape | Median CPU time | Derived value rate |
|---|---:|---:|
| SwiGLU, 4,096 values | 44.602 us | 91.84 million values/s |
| SwiGLU, 131,072 values | 1.512 ms | 86.66 million values/s |
| RoPE, 1 token x 32 heads x 128 | 1.904 us | 2.152 billion values/s |
| RoPE, 128 tokens x 32 heads x 128 | 187.894 us | 2.790 billion values/s |

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  --benchmark_filter='BM_Cpu(SwiGlu|Rope)' \
  --benchmark_min_warmup_time=0.1 \
  --benchmark_min_time=0.2s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

SwiGLU evaluates the stable SiLU branch plus multiplication, while RoPE consumes
already selected cosine/sine cache rows and rotates adjacent coordinate pairs.
Neither fixture includes cache generation or Tensor allocation. The unusually high
load produced 9–19% real-time coefficients of variation even though CPU-time
variation was lower; these values are retained as local baselines and should be
remeasured on an idle host before optimization conclusions.

## M10 contiguous CPU attention baseline — 2026-08-20

Environment:

- source: commit `c8ac20d`
- host: Apple M3 Pro, 12 CPU cores, 18 GB RAM
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, Float32
- Google Benchmark: 1.9.1
- output and `[H,Q,K]` score/probability workspace: preallocated and reused
- warmup: time-based 0.1 seconds per case
- measurement: time-based minimum 0.2 seconds per repetition, five repetitions;
  median CPU time reported
- one-minute load average at start: 15.64; affinity and CPU-frequency metadata were
  unavailable

| H x Q x K x D x DV | Median CPU time | Score workspace | Dot-product rate |
|---|---:|---:|---:|
| 8 x 1 x 128 x 64 x 64 | 76.371 us | 4,096 B | 3.433 GFLOP/s |
| 8 x 1 x 1,024 x 64 x 64 | 825.128 us | 32,768 B | 2.542 GFLOP/s |
| 3 x 17 x 29 x 37 x 41 | 46.284 us | 5,916 B | 4.985 GFLOP/s |
| 8 x 128 x 128 x 64 x 64 | 8.164 ms | 524,288 B | 4.110 GFLOP/s |

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  --benchmark_filter=BM_CpuContiguousAttention \
  --benchmark_min_warmup_time=0.1 \
  --benchmark_min_time=0.2s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

The reported rate counts only the multiply-add work in QK and probability-times-V,
`2*H*Q*K*(D+DV)`. Stable softmax and masking are included in elapsed time but not
that FLOP numerator. Decode latency scales with KV history in these two samples,
and prefill exposes the quadratic score workspace; neither observation is a speedup
claim. CUDA fixtures use identical shapes, ten warmups, and same-stream event
timing, but have not executed on a GPU.

## Timing rules

CPU microbenchmarks use Google Benchmark and a release build. Inputs are kept alive
with benchmark barriers where required, and setup/allocation is separated unless it
is the operation under study.

CUDA kernel timing uses warmed-up kernels and CUDA events recorded on the same
stream as the measured work. End-to-end host latency, transfer latency, and kernel
execution time are reported separately. A kernel launch measured without the
required synchronization is never reported as execution latency.

CUDA event fixtures now exist for M4 elementwise kernels, every M5
reduction/softmax variant, and both M6 SGEMM implementations. Their CUDA 12.6
compile validation is not a benchmark; result rows will be added only after
execution on a named NVIDIA GPU and driver.

M8's CUDA event-backed reclamation path likewise has correctness fixtures but no
published timing. A future comparison must use equivalent stream dependencies and
synchronization for direct `cudaMalloc/cudaFree`, Helix deferred reuse, and
`cudaMallocAsync`; a compile-only fixture or CPU completion mock is not a result.

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
