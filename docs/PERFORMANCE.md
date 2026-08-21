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
| M11 contiguous CPU KV cache | No earlier KV storage | Reserve per-sequence K/V tensors, append complete tokens, and copy a compact layer history into preallocated outputs | Lifecycle median CPU time: 26.898 us (2x2x64, capacity/append 128), 63.700 us (4x8x64, capacity 512/append 32), 7.853 us (odd 3x3x37, 29); layer-copy median: 1.595 us (128 KiB), 29.609 us (2 MiB), 613 ns (25,752 B) | Lifecycle includes sequence allocation, appends, and release. Layer copy excludes output allocation. This is a contiguous baseline under high host load, not evidence that paging or another layout is faster. |
| M12 paged CPU KV cache | Same-run M11 contiguous fixtures | Use a long-lived fixed-token pool, allocate logical blocks on append boundaries, and compact through block tables | Paged/contiguous lifecycle medians: 25.784/28.485 us, 18.016/64.400 us, 6.869/8.074 us. Paged/contiguous layer-copy medians: 3.684/1.568 us (128 KiB), 57.690/33.032 us (2 MiB), 540/554 ns (25,752 B). | Pool construction is outside paged lifecycle timing while per-sequence contiguous storage is inside; the 72% lifecycle reduction also uses 16x less assigned capacity. Paged copying is 2.35x and 1.75x slower for the large histories because it traverses 8 and 32 blocks. |
| M13 CPU attention value traversal | Dimension-major probability-times-V for contiguous and direct paged storage | Accumulate fixed 32-coordinate tiles while each V row/page is hot; retain both baselines | Contiguous speedup 1.31x–2.47x and paged speedup 2.36x–2.99x across four paired cases. With equal tiling, paged CPU time was 2.1%–6.7% higher. | The original apparent paged win was confounded by a general loop-locality improvement. Applying the same traversal to both paths isolates a modest end-to-end paging cost on this CPU; it is not GPU evidence. |

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

## M11 contiguous CPU KV-cache baseline — 2026-08-21

Environment:

- source: commit `ad7ebc2`
- host: Apple M3 Pro, 12 CPU cores, 18 GB RAM
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, Float32
- CMake 4.2.3 and Google Benchmark 1.9.1
- warmup: time-based 0.05 seconds per case
- measurement: time-based minimum 0.1 seconds per repetition, five repetitions;
  median CPU time reported
- concurrency: one benchmark thread; the cache itself uses one mutex
- host one-minute load average at start: 46.58; affinity and CPU-frequency
  metadata were unavailable

The lifecycle fixture times one sequence creation, a fixed number of whole-token
appends using already-created input tensors, and release. Capacity allocation is
included, but unused `Tensor::empty` capacity is neither initialized nor page-
touched by the fixture.

| Layers x heads x dimension | Capacity / appended tokens | Reserved bytes | Median CPU time per lifecycle | Appends/s within trace |
|---|---:|---:|---:|---:|
| 2 x 2 x 64 | 128 / 128 | 262,144 B | 26.898 us | 4.759 million/s |
| 4 x 8 x 64 | 512 / 32 | 8,388,608 B | 63.700 us | 502,355/s |
| 3 x 3 x 37 | 29 / 29 | 77,256 B | 7.853 us | 3.693 million/s |

The layer-copy fixture prefills one sequence outside the timed region and repeatedly
copies one layer's keys and values into preallocated compact outputs. The effective
rate counts copied payload bytes once; it is not a claim about total memory-bus
traffic.

| Layers x heads x dimension x tokens | Copied payload | Median CPU time | Effective copy rate |
|---|---:|---:|---:|
| 2 x 2 x 64 x 128 | 131,072 B | 1.595 us | 76.55 GiB/s |
| 4 x 8 x 64 x 512 | 2,097,152 B | 29.609 us | 65.96 GiB/s |
| 3 x 3 x 37 x 29 | 25,752 B | 613 ns | 39.11 GiB/s |

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  --benchmark_filter=BM_ContiguousKv \
  --benchmark_min_warmup_time=0.05 \
  --benchmark_min_time=0.1s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

These measurements establish only the contiguous CPU cost. The host was heavily
loaded, so they should be repeated on an idle machine before interpreting small
differences. The paired M12 run below repeats these contiguous cases in the same
process as their paged counterparts.

## M12 paged-versus-contiguous CPU KV baseline — 2026-08-21

Environment:

- source: commit `68e8d31`
- host: Apple M3 Pro, 12 CPU cores, 18 GB RAM
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, Float32
- CMake 4.2.3 and Google Benchmark 1.9.1
- warmup: time-based 0.05 seconds per case
- measurement: time-based minimum 0.1 seconds per repetition, five repetitions;
  median CPU time reported
- concurrency: one benchmark thread; both caches use one mutex
- host one-minute load average at start: 15.05; affinity and CPU-frequency
  metadata were unavailable

Both lifecycle fixtures time sequence creation, all whole-token appends, and
release with input tensors prepared outside the timed region. The contiguous cache
allocates and releases each sequence's declared-capacity tensors inside the loop.
The paged cache's global pool is constructed outside the loop, as it would be for a
long-lived runtime; the loop assigns and returns physical blocks. These are useful
steady-state runtime traces, but they do not isolate block-table overhead from the
different allocation boundary.

| Layers x heads x dimension; appended tokens | Contiguous declaration / assigned bytes | Paged block geometry / assigned pool bytes | Contiguous median | Paged median | Observed result |
|---|---:|---:|---:|---:|---|
| 2 x 2 x 64; 128 | capacity 128 / 262,144 B | 8 x 16 / 262,144 B | 28.485 us | 25.784 us | Paged was 9.5% lower CPU time with equal payload capacity. |
| 4 x 8 x 64; 32 | capacity 512 / 8,388,608 B | 2 x 16 / 524,288 B | 64.400 us | 18.016 us | Paged was 72.0% lower with 16x less assigned capacity in this over-reserved trace. |
| 3 x 3 x 37; 29 | capacity 29 / 77,256 B | 5 x 7 / 93,240 B | 8.074 us | 6.869 us | Paged was 14.9% lower despite six tail slots; pool allocation remains outside timing. |

No universal append-speed conclusion follows from these three shapes. They show
the intended lifetime model and that avoiding per-sequence maximum reservation can
matter. At equal exact capacities, block-table bookkeeping and the global-pool
construction boundary must be considered separately.

The layer-copy fixtures prefill outside timing and copy equal payloads into
preallocated compact outputs. Contiguous storage issues two copies per head. Paged
storage issues two copies per head per logical block.

| Layer history | Logical blocks | Contiguous median | Paged median | Observed result |
|---|---:|---:|---:|---|
| 2 heads x 128 tokens x 64; 131,072 B | 8 | 1.568 us | 3.684 us | Paged was 2.35x slower. |
| 8 heads x 512 tokens x 64; 2,097,152 B | 32 | 33.032 us | 57.690 us | Paged was 1.75x slower. |
| 3 heads x 29 tokens x 37; 25,752 B | 5 | 554 ns | 540 ns | The 2.5% difference is within observed variation; no winner is claimed. |

The unfavorable large-copy results are retained: materializing a contiguous
history erases paging's indirection benefit while adding copy calls. M13 paged
attention must read through the block table directly and be compared separately;
this benchmark does not predict its latency.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  '--benchmark_filter=BM_(Contiguous|Paged)Kv' \
  --benchmark_min_warmup_time=0.05 \
  --benchmark_min_time=0.1s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

No GPU storage or timing is represented by these CPU measurements.

## M13 direct paged-attention experiment — 2026-08-21

Environment:

- source: commit `aaf9d59` (paged baseline `26a9adf`, paged tiling `2977005`)
- host: Apple M3 Pro, 12 CPU cores, 18 GB RAM
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, Float32
- CMake 4.2.3 and Google Benchmark 1.9.1
- output and `[H,Q,K]` score/probability workspace: preallocated and reused
- paged pool construction, sequence append, and read-view acquisition: outside
  the timed region; the view lock remains held while timing
- warmup: time-based 0.05 seconds per case
- measurement: time-based minimum 0.1 seconds per repetition, five repetitions;
  median CPU time reported
- concurrency: one benchmark thread; attention is single-threaded
- one-minute host load average at start: 11.60; affinity and CPU-frequency metadata
  were unavailable

The original probability-times-V loop accumulated one output dimension at a time.
For paged storage, that also reacquired every page once per dimension. The tested
change uses a fixed `std::array<double, 32>` accumulator, loops over KV rows while a
32-coordinate value slice is contiguous and hot, and handles the final odd tile
without padding. It introduces no allocation and preserves the KV accumulation
order independently for every output coordinate. Both old paths remain in the
binary so baseline and changed implementations execute with identical setup.

| H x Q x K x D; page tokens/blocks | Contiguous major | Contiguous tiled | Contiguous speedup | Paged major | Paged tiled | Paged speedup | Tiled page overhead |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 x 1 x 128 x 64; 16/8 | 77.621 us | 42.299 us | 1.84x | 101.751 us | 43.188 us | 2.36x | 2.1% |
| 8 x 1 x 1,024 x 64; 16/64 | 813.689 us | 328.774 us | 2.47x | 889.556 us | 349.214 us | 2.55x | 6.2% |
| 3 x 17 x 29 x 37; 7/5 | 44.480 us | 34.060 us | 1.31x | 96.676 us | 35.490 us | 2.72x | 4.2% |
| 8 x 128 x 128 x 64; 16/8 | 8.192 ms | 3.544 ms | 2.31x | 11.300 ms | 3.780 ms | 2.99x | 6.7% |

CPU-time coefficients of variation for the tiled contiguous cases were 1.80%–
5.12%; tiled paged cases were 0.56%–3.07%. The reported page overhead is
`paged_tiled / contiguous_tiled - 1` using median CPU time. Q, K, and V have equal
head/value dimensions in all paired rows, and both implementations perform the
same double-accumulating math and stable softmax. The paged fixture fills the same
number of physical Float32 elements but obtains them through a sequence block
table; compact-copy time is absent from both sides.

This experiment corrected an important confound. Comparing optimized paged
attention only with the original contiguous dimension-major loop would have shown
a large but misleading paged win. Once both paths use the same value tile, direct
paging is modestly slower on this CPU. The comparison is still end-to-end rather
than a single-lookup microbenchmark: paged score traversal groups query work by
physical block, whereas contiguous score traversal groups all keys by query. The
two decode rows (`Q=1`) avoid that loop-order distinction and show 2.1% and 6.2%
overhead at eight and 64 pages respectively.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  --benchmark_filter='BM_Cpu(ContiguousAttention/(DimensionMajor|DimensionTiled)|PagedAttention/(DimensionMajor|DimensionTiled))' \
  --benchmark_min_warmup_time=0.05 \
  --benchmark_min_time=0.1s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

These are CPU layout/locality measurements. They say nothing about device block-
table fetches, warp grouping, coalescing, register pressure, occupancy, or GPU
latency. Those require a CUDA paged cache/kernel and physical NVIDIA execution.

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
