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
| M15 FIFO scheduler host trace | No earlier scheduler implementation | Schedule and complete deterministic mixed prefill/decode traces under token, running-request, and logical KV-block limits | Corrected median CPU time per complete trace: 48.777 us (16 requests/256 tokens) through 1.342 ms (128 requests/6,144 tokens) | A first fixture revision accidentally polled the O(N) statistics snapshot inside timing; those values were discarded. The corrected trace measures host policy and completion-vector work, not inference. |
| M16 continuous batching policy | Static cohorts on the same mixed-length request sequence | Refill finished slots between iterations and aggregate lifecycle metrics | Under an explicit constant-1-ms iteration model, continuous/static makespans were 48/132 ms and 92/260 ms (2.75x and 2.83x modeled rates). Median host control CPU times differed by 1.3% and 4.1%. | Modeled results demonstrate policy utilization only; they are not measured model throughput. Host differences are small relative to local variation and do not establish a CPU optimization. |
| M17 CPU Llama-style prefill and model load | No earlier complete model path | Compose the tested CPU kernels into full causal prefill and add checked `HELIXLM1` deserialization | Prefill medians: 97.942 us, 2.099 ms, and 34.104 ms for the three disclosed tiny configurations; checked in-memory load: 183.744 us, 640.046 us, and 3.967 ms | These establish single-threaded Float32 reference baselines, including forward temporaries or loader validation/allocation as specified. They do not demonstrate an optimization, GPU performance, disk-loading speed, or production model throughput. |
| M18 cached CPU next-position execution | Full-prefix recomputation for the identical prompt plus one input token | Reuse prefill K/V and execute only the new position | Cached/full-recompute median CPU times: 32.100/118.420 us, 89.580 us/2.113 ms, and 646.960 us/36.741 ms (3.69x, 23.59x, and 56.79x) | The growing advantage demonstrates avoided prefix model work in three tiny single-sequence CPU fixtures. It is not a production model, batched-runtime, tokenizer, or GPU result. |
| M18 contiguous-KV decode staging | Allocate a compact owning history snapshot, then copy it into the final history-plus-tail buffers | Copy the locked cache prefix directly into the final buffers | Two-stage/direct median CPU times: 858/466 ns, 1,161/634 ns, and 3,784/1,587 ns; reductions 45.7%, 45.4%, and 58.1% | Both paths include final-buffer allocation and current-tail writes. Direct copy removes two temporary tensors and one full history pass; the paired algorithm benchmark isolates this change from model math. |
| M19 synchronous CPU runtime prefill | Execute every scheduled prompt token through the one-token paged decode path | Execute a complete newly admitted prompt once and bulk-append its layer-major K/V pages | Sequential/full-prompt median CPU trace times: 461.833/224.800 us, 2.616/1.675 ms, and 11.745/9.890 ms; reductions 51.3%, 36.0%, and 15.8% | Full prefill wins all paired tiny traces, but the benefit narrows as decode dominates. The timed region is real serial CPU model/scheduler/cache execution, not GPU or fused-batch throughput. |
| M19 paged model decode staging | Copy all cached pages into two contiguous history-plus-current K/V tensors per layer | Traverse leased block-table spans plus an uncommitted current K/V tail | Isolated compact/direct medians: 1,609/1,239 ns, 2,359/2,045 ns, 15.667/11.324 us, and 112.007/88.662 us; reductions 23.0%, 13.3%, 27.7%, and 20.8% | Direct traversal eliminates 2 KiB–1 MiB of combined-K/V payload per layer call. Complete tiny trace medians improved only 0.08%–6.23%, so the end-to-end impact is explicitly modest. |
| M22 block-aligned prefix sharing | Create a child sequence and bulk-copy the already-computed prefix K/V payload into newly assigned pages | Copy only block IDs and increment physical reference counts | Duplicate/shared medians: 916/83.1 ns for 32 KiB, 3.180 us/93.3 ns for 93,240 B, and 29.598 us/92.3 ns for 1 MiB (11.02x, 34.08x, and 320.67x) | The paired lifecycle isolates the page-sharing mechanism. Assigned child payload falls to zero, but the fixed pool is pre-reserved and content lookup, hashing, eviction, model execution, and GPU work are excluded. |
| M22 exact prefix content index | Linear exact longest-prefix scan retained in the same binary | Incremental token digest plus namespace/length hash probes and full-vector collision verification | At 1/64/256/1,024 entries, indexed versus linear hot-hit medians were 136/81.2 ns, 164/172 ns, 153/441 ns, and 165/1,529 ns; miss medians were 31.8/7.61 ns, 34.4/115 ns, 34.0/404 ns, and 33.1/1,961 ns | Indexing loses on tiny caches but bounds entry-count scaling in this fixture. Exact token comparison preserves correctness; the linear baseline remains benchmarkable and no machine-specific hybrid cutoff was encoded. |
| M22 hot CPU runtime prefix reuse | Full vectorized prefill for every request after an equivalent untimed warm request | Exact lookup, page fork, four-token suffix execution, and duplicate-anchor refresh | Disabled/enabled median CPU trace times: 434.232/732.500 us (8-token prompt), 3.375/1.877 ms (16), and 28.370/4.695 ms (32): 68.7% regression, 1.80x speedup, and 6.04x speedup | Reuse reduces executed prompt tokens but fixed lookup/fork cost and sequential suffix execution lose on the shortest trace. The feature remains opt-in; these are tiny serial CPU reference workloads, not GPU/production serving results. |

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

## M15 FIFO scheduler host-policy baseline — 2026-08-21

Environment:

- source: scheduler commit `7b50dbc`; corrected fixture commit `f17ce74`
- host: Apple M3 Pro, 12 CPU cores, 18 GB RAM
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, warnings as errors
- CMake 4.2.3 and Google Benchmark 1.9.1
- request construction and submission: outside the timed region
- timed work: every `schedule_iteration`, completion-vector construction, and
  `complete_iteration` call until the trace finishes
- completion detection: derived from each returned retirement report; no statistics
  snapshot or O(N) request-history scan occurs inside the timed loop
- excluded work: model execution, token selection, physical KV allocation/data,
  device transfer, and CUDA execution
- synthetic KV capacity: enough blocks for every request's full prompt and output;
  reported batch reservations are deducted and terminal releases returned
- logical time: advanced deterministically by one microsecond around each
  completion; it is metadata input and not used as benchmark duration
- warmup: time-based 0.05 seconds per case
- measurement: time-based minimum 0.1 seconds per repetition, five repetitions;
  median CPU time reported
- concurrency: one benchmark thread and one scheduler owner
- one-minute host load average at start: 7.90; affinity and CPU-frequency metadata
  were unavailable

| Requests | Prompt/output tokens each | Max running | Token budget | Block tokens | Scheduler iterations | Median CPU time/trace | CPU-time CV | Logical transitions/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 8 / 8 | 8 | 64 | 16 | 18 | 48.777 us | 2.38% | 5.248 million |
| 64 | 32 / 16 | 16 | 128 | 16 | 72 | 367.201 us | 3.36% | 8.366 million |
| 64 | 128 / 32 | 32 | 512 | 16 | 74 | 698.269 us | 1.37% | 14.665 million |
| 128 | 17 / 31 | 32 | 256 | 7 | 130 | 1.342 ms | 3.08% | 4.579 million |

“Logical transitions/s” divides prompt-plus-generated token state transitions by
the measured host CPU time. It is useful only for comparing scheduler policy
implementations on the same trace. It is not tokens generated per second: the
fixture performs no tensor operation, attention, cache write, sampling, or GPU
work. The non-monotonic rate reflects differing active-set sizes, prompt chunking,
decode iteration counts, and container/allocation work; no optimization conclusion
is drawn from this first baseline.

The fixture initially used `scheduler.stats().finished_count` as its loop
condition. `stats()` walks retained request history to classify every state, so
that observer added unrelated O(N) work to every measured iteration. The values
recorded before `f17ce74` are discarded rather than presented as a scheduler
regression or optimization. The corrected fixture counts finished requests from
the already-required `CompletionReport`; the table above is the replacement
measurement.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  --benchmark_filter=BM_FifoSchedulerTrace \
  --benchmark_min_warmup_time=0.05 \
  --benchmark_min_time=0.1s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

## M16 static-versus-continuous scheduler experiment — 2026-08-21

Environment and measured host scope:

- source: commit `f17ce74`
- host/OS/compiler/build: Apple M3 Pro (12 CPU cores, 18 GB RAM), macOS
  26.0.1 arm64, Apple Clang 17.0.0, Release, `HELIX_ENABLE_CUDA=OFF`, warnings as
  errors
- CMake 4.2.3 and Google Benchmark 1.9.1
- measured host work: scheduler/request construction, prompt construction and
  submission, all scheduling/completion decisions and completion vectors, and the
  final metrics snapshot; only the immutable workload specification is outside
  timing
- warmup: time-based 0.05 seconds per case
- measurement: time-based minimum 0.1 seconds per repetition, five repetitions;
  median CPU time reported
- concurrency: one benchmark thread and one scheduler owner
- one-minute host load average at start: 7.90; affinity and CPU-frequency metadata
  were unavailable

The same fixture also advances request timestamps by exactly 1 ms per completed
scheduler iteration. This is a declared policy model, not elapsed benchmark time.
It assumes every batch has equal cost regardless of resident count, prefill/decode
mix, prompt chunk, tensor shape, or KV length. No model, attention, cache write,
device transfer, or GPU kernel executes. Enough logical KV blocks are provisioned
to remove memory pressure as a confound, and all requests arrive at modeled time
zero.

Workloads are deliberately mixed-length. In workload S, 32 requests have prompt
length 8, concurrency 8, token budget 64, and 16-token KV blocks; the first request
of each eight-request group emits 32 tokens and the other seven emit 4. In workload
L, 128 requests have prompt length 17, concurrency 32, token budget 544, and
7-token blocks; each group leader emits 64 tokens and the other 31 emit 8. Static
policy admits the next fixed cohort only after every request in the prior cohort
finishes. Continuous policy queues all requests and refills each terminal slot on
the next iteration.

Modeled utilization/rate results:

| Workload | Policy | Iterations / modeled ms | Modeled requests/s | Modeled generated tokens/s |
|---|---|---:|---:|---:|
| S: 32 requests, max 8 | Continuous | 48 | 666.667 | 5,000.000 |
| S: 32 requests, max 8 | Static cohorts | 132 | 242.424 | 1,818.182 |
| L: 128 requests, max 32 | Continuous | 92 | 1,391.304 | 13,565.217 |
| L: 128 requests, max 32 | Static cohorts | 260 | 492.308 | 4,800.000 |

Modeled latency distributions (milliseconds):

| Workload/policy | Wait mean / p50 / p95 | TTFT p50 / p95 | Request latency p50 / p95 | Mean inter-token |
|---|---:|---:|---:|---:|
| S continuous | 9.375 / 10 / 22.25 | 12 / 24.25 | 15 / 40.25 | 1 |
| S static | 49.5 / 49.5 / 99 | 51.5 / 101 | 68.5 / 104 | 1 |
| L continuous | 14.203 / 18 / 27 | 20 / 29 | 27 / 45 | 1 |
| L static | 97.5 / 97.5 / 195 | 99.5 / 197 | 134.5 / 204 | 1 |

Continuous batching therefore produces 2.75x and 2.83x the modeled request/token
rate for these two intentionally imbalanced arrival sequences. That conclusion is
conditional on the constant-cost iteration model. It demonstrates that refilling
slots removes static-cohort bubbles; it does not predict the speedup of a real
transformer, where prefill and decode have different costs and batch efficiency is
shape/hardware dependent.

Measured host control-path results:

| Workload | Continuous median CPU time | Static median CPU time | CPU-time CVs |
|---|---:|---:|---:|
| S | 105.937 us | 107.290 us | 1.28% / 2.33% |
| L | 491.558 us | 512.741 us | 2.94% / 1.93% |

Continuous host time was 1.3% and 4.1% lower in this run, but the implementations
perform different iteration/container traces and the gaps are small relative to
local variation. No CPU optimization claim is made. The modeled policy outcome is
deterministic; the host timing remains a local control-path baseline.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  --benchmark_filter='BM_(ContinuousBatchingPolicy|StaticCohortPolicy)' \
  --benchmark_min_warmup_time=0.05 \
  --benchmark_min_time=0.1s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

## M17 CPU model baseline — 2026-08-23

Environment:

- source: implementation and benchmark commit `987c3b2`;
- host: Apple M3 Pro, 12 CPU cores (6 performance, 6 efficiency), 18 GB RAM;
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, warnings as errors, Float32;
- CMake 4.2.3 and Google Benchmark 1.9.1;
- concurrency: one benchmark thread; CPU reference kernels are single-threaded;
- warmup: one complete untimed operation after fixture setup for each benchmark
  invocation;
- measurement: minimum 0.1 seconds, seven repetitions, median CPU time reported;
- one-minute load average at start: 9.08; affinity and CPU-frequency metadata were
  unavailable on this host.

The full-prefill fixture constructs deterministic model weights and token IDs
outside timing. Each measured iteration calls `CpuLlamaModel::forward` and includes
embedding lookup, RoPE-cache construction, all layer math, causal attention,
token/head repacking, every temporary Tensor allocation/destruction, final norm,
and the full `[T,V]` logits projection. It excludes model construction, weight
initialization, token construction, and diagnostic trace retention.

| H / I / L / heads / T / V | Parameters / bytes | Iterations per repetition | Median CPU time | CPU-time CV | Prompt tokens/s |
|---|---:|---:|---:|---:|---:|
| 32 / 64 / 2 / 4 / 7 / 128 | 28,832 / 115,328 B | 1,494 | 97.942 us | 2.15% | 71,470.6 |
| 64 / 128 / 2 / 4 / 31 / 256 | 115,008 / 460,032 B | 68 | 2.099 ms | 2.95% | 14,768.7 |
| 128 / 256 / 4 / 8 / 63 / 512 | 787,584 / 3,150,336 B | 4 | 34.104 ms | 4.52% | 1,847.3 |

“Prompt tokens/s” is `T / full-prefill latency` for this one-sequence fixture. It
is not autoregressive decode throughput: no generation loop, KV reuse, scheduler,
batch, tokenizer, disk I/O, or CUDA operation runs. The models are deterministic
synthetic weights and are orders of magnitude smaller than production LLMs. The
falling rate reflects changing model dimensions and quadratic attention together;
the fixture does not isolate an individual bottleneck.

The load fixture serializes deterministic weights and performs one load outside
timing. Each measured iteration constructs an in-memory input stream, validates
the fixed header and complete tensor directory, verifies metadata and payload
checksums, checks zero padding and finite values, decodes little-endian Float32
payloads into aligned Tensor allocations, assembles move-only layer weights, and
validates the final model. Serialization and filesystem I/O are excluded.

| H / I / L / heads / T / V | File bytes | Iterations per repetition | Median CPU time | CPU-time CV |
|---|---:|---:|---:|---:|
| 32 / 64 / 2 / 4 / 7 / 128 | 117,184 B | 820 | 183.744 us | 2.86% |
| 64 / 128 / 2 / 4 / 31 / 256 | 461,888 B | 216 | 640.046 us | 6.01% |
| 128 / 256 / 4 / 8 / 63 / 512 | 3,153,600 B | 37 | 3.967 ms | 5.07% |

The loader values are memory-resident parsing/allocation baselines, not storage
throughput. They intentionally include portable byte decoding and two FNV-1a
passes (one metadata, one per payload during decoding). No alternate loader exists
yet, so there is no optimization result or checksum-cost conclusion.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  '--benchmark_filter=BM_(CpuLlamaFullPrefill|LlamaModelLoadFromMemory)' \
  --benchmark_min_time=0.1s \
  --benchmark_repetitions=7 \
  --benchmark_report_aggregates_only=false
```

An earlier fixture revision paused Google Benchmark timing before entering the
state iteration loop; its nonsensical uptime-scale real-time values were discarded.
The recorded fixture enters the loop first, pauses only to initialize and warm up
once, then resumes before its first measured operation. This correction is
methodology repair, not a runtime speedup.

## M18 CPU cached decode and staging optimization — 2026-08-23

Environment:

- source: implementation commit `c515d6d` plus the explicit staging warmup in the
  benchmark fixture;
- host: Apple M3 Pro, 12 CPU cores (6 performance, 6 efficiency), 18 GB RAM;
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, warnings as errors, Float32;
- CMake 4.2.3 and Google Benchmark 1.9.1;
- concurrency: one benchmark thread; model and reference kernels are
  single-threaded;
- timing: Google Benchmark process CPU time, with setup excluded as described
  below; benchmark order was randomly interleaved across repetitions;
- CPU affinity and frequency metadata were unavailable. The reported synthetic
  24 MHz metadata is a failed `sysctl` estimate and must not be used.

### Cached decode versus full-prefix recomputation

Both variants compute logits for the same next input position. The cached fixture
constructs a new deterministic model/session and performs prompt prefill while
timing is paused for every measured operation; its timed region includes one-token
model math, RoPE at the absolute position, locked cache-prefix copies, temporary
Tensor allocation/destruction, final cache append, and `[1,V]` logits. It also
executes one complete prefill/decode warmup before the first measured operation.

The recomputation fixture constructs one deterministic model outside timing,
warms up one complete forward call, and then executes `forward` over the prompt
plus decode input in every timed operation. Token/model construction and result
comparison are outside both timed regions. Each case uses exactly 50 timed
operations per repetition and five repetitions; median CPU time and aggregate CV
are reported.

| H / I / L / heads / prompt / total / V | Parameters | Cached decode median | Cached CV | Full recompute median | Full CV | Recompute / cached |
|---|---:|---:|---:|---:|---:|---:|
| 32 / 64 / 2 / 4 / 7 / 8 / 128 | 28,832 | 32.100 us | 7.47% | 118.420 us | 5.21% | 3.69x |
| 64 / 128 / 2 / 4 / 31 / 32 / 256 | 115,008 | 89.580 us | 2.85% | 2.113 ms | 5.75% | 23.59x |
| 128 / 256 / 4 / 8 / 63 / 64 / 512 | 787,584 | 646.960 us | 2.47% | 36.741 ms | 1.94% | 56.79x |

The comparison demonstrates the expected benefit of not recomputing prior-token
embedding, projections, MLPs, and causal attention. It does not isolate KV traffic
from the other saved work, and the increasing ratio combines larger prefixes with
changing model dimensions. Prompt prefill is deliberately excluded, so these are
steady next-position costs, not request latency or amortized tokens/s. The fixture
uses tiny deterministic synthetic weights, one sequence, greedy-independent
logits, and no scheduler, tokenizer, BLAS, or GPU.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  '--benchmark_filter=BM_CpuLlamaNextPosition/(CachedKvDecode|FullPrefixRecompute)' \
  --benchmark_min_time=50x \
  --benchmark_repetitions=5 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true
```

### Direct cache-prefix staging

The original cached-decode baseline called `retrieve_layer`, allocating and
copying compact key/value histories, then allocated the final key/value buffers
and copied history a second time before writing the current token. The optimized
path allocates only the final pair and uses `copy_layer_prefix_out` to copy cached
rows directly into their prefix. Both benchmark variants include final-buffer
allocation, current-token tail copies, cache locking, and destruction. The
two-stage variant additionally includes its two snapshot allocations and extra
history pass. Cache construction/fill and deterministic token construction are
outside timing; each variant runs once as an explicit warmup.

Each case uses exactly 100,000 timed operations per repetition, ten repetitions,
and random interleaving. `History bytes` counts keys plus values for one layer.

| Heads / D / history | History bytes | Two-stage median / CV | Direct median / CV | Median reduction | Speedup |
|---|---:|---:|---:|---:|---:|
| 4 / 8 / 7 | 1,792 B | 858 ns / 4.78% | 466 ns / 2.76% | 45.7% | 1.84x |
| 4 / 16 / 31 | 15,872 B | 1,161 ns / 3.56% | 634 ns / 4.69% | 45.4% | 1.83x |
| 8 / 16 / 63 | 64,512 B | 3,784 ns / 3.32% | 1,587 ns / 6.16% | 58.1% | 2.38x |

This paired result supports retaining the direct-prefix API. It does not imply the
same percentage for end-to-end decode: model math dominates there, and a separate
before/after process comparison was too sensitive to local system load to use as
the optimization claim.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  '--benchmark_filter=BM_ContiguousKvDecodeStaging' \
  --benchmark_min_time=100000x \
  --benchmark_repetitions=10 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true
```

## M19 synchronous CPU runtime prefill optimization — 2026-08-23

Environment:

- source: commit `2a6b365` (including implementation commit `8086257`);
- host: Apple M3 Pro, 12 CPU cores (6 performance, 6 efficiency), 18 GB RAM;
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, warnings as errors, Float32;
- CMake 4.2.3 and Google Benchmark 1.9.1;
- concurrency: one benchmark thread; requests share one paged pool, but scheduled
  model work executes serially through the single-threaded CPU kernels;
- timing: Google Benchmark process CPU time, 30 complete request traces per
  repetition, nine randomly interleaved repetitions;
- warmup: one complete trace per mode/workload before its first measured trace;
- CPU affinity and frequency metadata were unavailable. The displayed synthetic
  24 MHz metadata is a failed `sysctl` estimate and is not used.

Each trace constructs deterministic weights, the engine, paged pool, and all
requests while timing is paused. The timed region repeatedly calls
`run_iteration` until the engine's O(1) live-queue query is false. It includes
scheduler policy, Float32 model temporaries/math, paged prefix reads and appends,
greedy selection, logical completion, and physical reclamation. Engine destruction,
request submission, the final ownership/statistics validation snapshot, and model
construction are excluded. Scheduler timestamps advance by a declared synthetic
one microsecond per iteration solely to satisfy monotonic lifecycle accounting;
they do not delay or scale the measured CPU time.

All workloads use vocabulary 128, block size four, token budget
`active_requests * prompt_tokens`, and sufficient pool capacity for the maximum
resident prefixes. Every group of `active_requests` has one long request; other
requests generate `ceil(long_output/4)` tokens. `Logical tokens` counts prompt plus
generated scheduler work, not just generated output. Peak K/V bytes are actual
paged-pool assigned bytes, not the larger reserved pool.

| H / I / L / heads | Requests / active | Prompt / long / short output | Logical tokens / runtime iterations | Peak K/V bytes | Sequential median / CPU CV | Full-prompt median / CPU CV | Reduction / speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 / 32 / 1 / 4 | 8 / 4 | 4 / 4 / 1 | 46 / 7 | 3,072 B | 461.833 us / 3.86% | 224.800 us / 7.30% | 51.3% / 2.05x |
| 32 / 64 / 2 / 4 | 8 / 4 | 8 / 8 / 2 | 92 / 12 | 26,624 B | 2.616 ms / 3.72% | 1.675 ms / 4.26% | 36.0% / 1.56x |
| 64 / 128 / 2 / 4 | 16 / 8 | 8 / 8 / 2 | 172 / 12 | 102,400 B | 11.745 ms / 5.12% | 9.890 ms / 3.85% | 15.8% / 1.19x |

The corresponding optimized logical-work rates were 204.626k, 54.920k, and
17.392k tokens/s; successful request rates were 35.587k, 4.776k, and 1.618k
requests/s. These are tiny synthetic, serial CPU reference-runtime rates. They are
not GPU throughput, public-model throughput, or evidence that executing multiple
work descriptions serially provides tensor batching speedup.

The retained `SequentialTokenPrefill` mode invokes paged one-token decode for each
prompt position. The `FullPromptPrefill` mode performs one full causal prefill for
a complete new prompt, retains its last logits row, and transactionally bulk-
appends layer-major K/V. Both modes use identical decode, scheduler, request, and
pool logic. This historical prefill-only fixture now pins the retained compact-copy
decode mode so the later direct-decode default cannot contaminate the comparison.
The smaller relative gain at the largest case is an honest consequence of decode
occupying more of the trace; the optimization changes prompt work only.

The first fixture revision polled `SchedulerStats` (an O(number of retained
requests) reporting snapshot) through `has_live_requests` on every iteration and
included one final stats snapshot in timing. Those provisional values were
discarded. The recorded version exposes an O(1) waiting/running query and pauses
timing before final validation. This is a benchmark-methodology correction in
addition to a useful production API property; it is not included as a claimed
speedup.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  '--benchmark_filter=BM_CpuRuntimeContinuousTrace' \
  --benchmark_min_time=30x \
  --benchmark_repetitions=9 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true
```

## M19 direct paged-decode traversal — 2026-08-23

Hypothesis: paged model decode was paying for the cache layout twice. Each layer
allocated two `[H,history+1,D]` buffers, copied cached page spans into them, wrote
current K/V into the tail, and only then ran contiguous attention. Since M13
already had checked direct page traversal, accepting current K/V as an uncommitted
causal tail should remove that staging without changing model or cache ownership.

Environment:

- source: commit `c9b151c`;
- host: Apple M3 Pro, 12 CPU cores (6 performance, 6 efficiency), 18 GB RAM;
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, warnings as errors, Float32;
- CMake 4.2.3 and Google Benchmark 1.9.1;
- one benchmark thread; CPU affinity/frequency metadata unavailable, so the
  displayed synthetic 24 MHz estimate is ignored;
- load averages at command start: 4.53/4.71/5.83 for the isolated fixture and
  5.04/6.05/6.60 for the complete runtime fixture.

The isolated fixture owns a one-layer paged cache populated before timing. Each
mode receives the same deterministic `[H,1,D]` query/current K/V. Both allocating
attention APIs create output and `[H,1,history+1]` score workspace. `CompactCopy`
additionally allocates two combined K/V tensors, copies the locked history, writes
the current row, and calls contiguous attention. `DirectBlockTable` acquires a
move-only read lease, traverses cached page spans, and consumes current K/V as the
tail. One complete call per mode/shape warms allocation and code paths; each
reported repetition then measures exactly 1,000 calls using process CPU time, with
15 repetitions randomly interleaved.

| H / D / history / page | Compact median / CPU CV | Direct median / CPU CV | Combined K/V payload removed | Reduction / speedup |
|---|---:|---:|---:|---:|
| 4 / 16 / 3 / 4 | 1,609 ns / 7.56% | 1,239 ns / 6.30% | 2,048 B | 23.0% / 1.299x |
| 4 / 16 / 15 / 4 | 2,359 ns / 7.01% | 2,045 ns / 6.74% | 8,192 B | 13.3% / 1.154x |
| 8 / 32 / 63 / 16 | 15.667 us / 5.02% | 11.324 us / 5.36% | 131,072 B | 27.7% / 1.384x |
| 8 / 64 / 255 / 16 | 112.007 us / 5.57% | 88.662 us / 4.95% | 1,048,576 B | 20.8% / 1.263x |

The removed payload is the exact logical size of two combined K/V tensors,
`2 * H * (history + 1) * D * sizeof(float)`. It is not a malloc-overhead or
whole-process peak-memory measurement. Direct traversal still allocates output and
score workspace and holds the cache-wide lease throughout attention.

The second fixture measures complete full-prefill runtime traces. It preserves the
M19 request/model shapes, scheduler, global paged pool, generation lengths, greedy
selection, completion, and reclamation; only `PagedDecodeMode` changes. Setup,
submission, destruction, and final statistics are excluded. Each mode receives one
complete warmup trace, followed by 100 traces per repetition and 15 randomly
interleaved repetitions.

| H / I / L / heads | Requests / active | Prompt / long / short output | Compact median / CPU CV | Direct median / CPU CV | Reduction / speedup |
|---|---:|---:|---:|---:|---:|
| 16 / 32 / 1 / 4 | 8 / 4 | 4 / 4 / 1 | 230.460 us / 5.64% | 221.300 us / 5.50% | 3.97% / 1.041x |
| 32 / 64 / 2 / 4 | 8 / 4 | 8 / 8 / 2 | 1.640 ms / 5.78% | 1.538 ms / 4.95% | 6.23% / 1.066x |
| 64 / 128 / 2 / 4 | 16 / 8 | 8 / 8 / 2 | 9.791 ms / 2.84% | 9.783 ms / 4.02% | 0.08% / 1.001x |

The isolated mechanism improves consistently, while full traces spend most time
in GEMM, normalization, MLP, scheduler, and other model work. In particular, the
largest end-to-end result is indistinguishable from a tie at this noise level. The
direct path is the default because it removes deterministic history-sized
temporaries, matches the compact/full-recompute oracles, wins the isolated paired
workload, and did not regress these longer trace samples—not because a broad
end-to-end speedup has been established.

Commands:

```bash
./build-release/benchmarks/helix_benchmarks \
  '--benchmark_filter=BM_CpuPagedDecodeAttention' \
  --benchmark_min_time=1000x \
  --benchmark_repetitions=15 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true

./build-release/benchmarks/helix_benchmarks \
  '--benchmark_filter=BM_CpuRuntimePagedDecodeTrace' \
  --benchmark_min_time=100x \
  --benchmark_repetitions=15 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true
```

## M22 block-aligned prefix sharing — 2026-08-23

Hypothesis: once a K/V prefix already occupies complete physical pages, creating
another logical sequence by copying that immutable payload wastes both bandwidth
and scarce assigned page capacity. Copying just the page IDs and incrementing
reference counts should make fork cost scale with the number of pages rather than
`layers * heads * tokens * head_dimension`, provided all shared pages are full.

Environment:

- source: commit `56d0cd0`;
- host: Apple M3 Pro, 12 CPU cores (6 performance, 6 efficiency), 18 GB RAM;
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, warnings as errors, Float32;
- CMake 4.2.3 and Google Benchmark 1.9.1;
- one benchmark thread; CPU affinity/frequency metadata unavailable, so the
  displayed synthetic 24 MHz estimate is ignored;
- load averages at command start: 10.22/6.97/6.12.

Each mode constructs a fixed pool with twice the prefix's page count, deterministic
layer-major K/V, and one populated source while timing is excluded. One complete
child lifecycle warms each mode. Every measured operation then either creates an
empty child, bulk-appends the same payload, checks length, and releases it
(`DuplicatePayload`), or forks the source's complete block table, checks length,
and releases it (`SharedBlocks`). Both include sequence-map/table metadata, the
cache mutex, and final reference/reclamation bookkeeping. Each repetition executes
exactly 10,000 lifecycles; 15 repetitions are randomly interleaved, and median
process CPU time plus aggregate CPU-time CV are reported.

| L / heads / D / block / prefix | Prefix K/V payload | Duplicate median / CPU CV | Shared median / CPU CV | Reduction / speedup |
|---|---:|---:|---:|---:|
| 2 / 2 / 32 / 8 / 32 | 32,768 B | 916 ns / 2.07% | 83.1 ns / 5.70% | 90.9% / 11.02x |
| 3 / 3 / 37 / 7 / 35 | 93,240 B | 3.180 us / 4.53% | 93.3 ns / 4.37% | 97.1% / 34.08x |
| 4 / 4 / 64 / 16 / 128 | 1,048,576 B | 29.598 us / 3.28% | 92.3 ns / 4.41% | 99.7% / 320.67x |

The payload column is exact `prefix_tokens * bytes_per_token`, including keys and
values for every layer/head. The duplicate assigns and writes that many additional
bytes; the shared child assigns zero new payload bytes and adds four, five, or
eight block-table references. Because `PagedKvCache` reserves its entire pool at
construction in both modes, this is saved assigned capacity—not reduced process
RSS or pool reservation. The widening speedup mostly reflects payload-copy growth
against small metadata work; it does not establish an end-to-end request speedup.

The fixture deliberately excludes prompt hashing, collision verification,
longest-prefix search, anchor retention, eviction, model execution, and scheduling.
It therefore validates the M22 ownership primitive and its copy baseline, not a
complete prefix cache. No CUDA code ran.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  '--benchmark_filter=BM_PagedKvPrefixForkLifecycle' \
  --benchmark_min_time=10000x \
  --benchmark_repetitions=15 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true
```

## M22 exact prefix-index scaling — 2026-08-23

Hypothesis: scanning every retained content entry makes both misses and longest-
prefix hits grow with cache population. An incremental digest should let lookup
scan the query once and probe only candidates at block-aligned lengths. Full token
comparison must remain mandatory so a digest collision cannot establish cache
identity.

Environment:

- source: commit `2d5202d` (production index from `888dd30`, retained linear
  benchmark from `dc5aa5e`);
- host: Apple M3 Pro, 12 CPU cores (6 performance, 6 efficiency), 18 GB RAM;
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, warnings as errors, Float32;
- CMake 4.2.3 and Google Benchmark 1.9.1;
- one benchmark thread; affinity/frequency metadata unavailable;
- load averages at the main scaling run start: 6.00/7.05/7.67;
- warmup: one lookup lifecycle before each hot case and one miss before each miss
  case;
- measurement: time-based minimum 0.1 seconds per case per repetition, 15 random-
  interleaved repetitions; median process CPU time reported.

Every entry is an exact eight-token key occupying two four-token pages in a
one-layer/one-head/one-dimension cache. Entries differ in their first token. The
hot query matches one exact entry and includes lookup, anchor fork, MRU update, and
child release. The miss differs in its first token and measures lookup only. Pool
and entry construction are outside timing. `Linear` is the original mutex-
protected exact LRU scan compiled into the same executable; `Indexed` is
`PagedPrefixCache`. A direct fork/release floor measured 69.5 ns median in the
same run.

| Entries | Linear hot lifecycle | Indexed hot lifecycle | Linear miss | Indexed miss |
|---:|---:|---:|---:|---:|
| 1 | 81.2 ns | 136 ns | 7.61 ns | 31.8 ns |
| 16 | 120 ns | 169 ns | 29.0 ns | 41.5 ns |
| 64 | 172 ns | 164 ns | 115 ns | 34.4 ns |
| 256 | 441 ns | 153 ns | 404 ns | 34.0 ns |
| 1,024 | 1,529 ns | 165 ns | 1,961 ns | 33.1 ns |

The negative small-cache result is real: at one entry, indexing adds 67.5% to the
hot lifecycle and makes the miss 4.18x slower. The indexed hot path is effectively
tied at 64 entries, then measures 2.88x and 9.27x faster at 256 and 1,024 entries.
Indexed misses measure 3.34x, 11.88x, and 59.2x faster at 64, 256, and 1,024.
A targeted 24/32/48-entry run placed the miss crossover between 16 and 24 entries,
while hot lookup remained slower through 48 and crossed by 64 on this host.

The production path stays indexed rather than embedding a benchmark-host-specific
hybrid threshold. Its expected lookup work depends on query length/aligned
boundaries and collision-bucket size instead of total entries. The linear
implementation remains only in the benchmark so a future trie, radix index, or
hybrid proposal must beat both sides of this crossover. These numbers cover CPU
metadata and page-reference operations only; no model execution or CUDA work ran.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  '--benchmark_filter=BM_PagedPrefix(DirectForkLifecycle|CacheHotHitLifecycle|CacheMiss|LinearHotHitLifecycle|LinearMiss)' \
  --benchmark_min_time=0.1s \
  --benchmark_repetitions=15 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true
```

## M22 hot CPU runtime prefix reuse — 2026-08-23

Hypothesis: once an aligned prompt prefix has a persistent K/V anchor, forking its
pages and executing only the uncached suffix should reduce complete request-trace
CPU time when avoided model work exceeds lookup, reference, and incremental-suffix
overhead. A paired short-prompt case is required to expose the crossover rather
than assuming reuse always wins.

Environment:

- source: commit `7840546` (runtime integration from `fbc6b20`);
- host: Apple M3 Pro, 12 CPU cores (6 performance, 6 efficiency), 18 GB RAM;
- OS/compiler/build: macOS 26.0.1 arm64, Apple Clang 17.0.0, Release,
  `HELIX_ENABLE_CUDA=OFF`, warnings as errors, Float32;
- CMake 4.2.3 and Google Benchmark 1.9.1;
- one benchmark thread; affinity/frequency metadata unavailable;
- load averages at command start: 6.75/6.37/6.64;
- warmup: construct one engine and complete one request outside timing for every
  measured trace; this populates the prefix only in the enabled variant while
  warming equivalent model work in both;
- measurement: time-based minimum 0.1 seconds per case per repetition, 15 random-
  interleaved repetitions; median process CPU time and CPU-time CV reported.

Each measured trace submits and completes 16 sequential requests on the warmed
engine. Vocabulary is 128, K/V pages hold four tokens, every request emits one
length-limited output, and all prompts share the maximum reusable aligned prefix
but vary in their final page. Engine/model construction, weight generation, warmup,
statistics validation, and destruction are outside timing. Timed work includes
request submission, exact lookup, scheduling, serial CPU model execution, paged
append/read/release, request completion, and prefix duplicate refresh. Both modes
use full-prompt prefill when starting at offset zero and direct block-table decode;
a hit necessarily executes its four-token suffix through the sequential path.

| H / I / L / heads | Prompt / shared | Disabled median / CPU CV | Enabled median / CPU CV | Executed prefill tokens (off/on) | Result |
|---|---:|---:|---:|---:|---:|
| 16 / 32 / 1 / 4 | 8 / 4 | 434.232 us / 2.79% | 732.500 us / 3.79% | 128 / 64 | 68.7% slower |
| 32 / 64 / 2 / 4 | 16 / 12 | 3.375 ms / 1.44% | 1.877 ms / 2.49% | 256 / 64 | 1.80x faster |
| 64 / 128 / 2 / 4 | 32 / 28 | 28.370 ms / 2.06% | 4.695 ms / 2.12% | 512 / 64 | 6.04x faster |

All variants complete 32 runtime iterations (one prefill and one terminal decode
per measured request). Peak physical K/V payload was unchanged at 1,024 B, 8,192
B, and 32,768 B because both variants reach the same full-prompt page footprint;
sharing reduces recomputation and duplicate page assignment, not the fixed pool's
reservation. The cache-enabled scheduler reports exactly 64, 192, and 448 reused
prompt tokens across the 16 measured requests.

The 8-token regression matters: even after eliminating half of prompt tokens,
hashing, locking, page forks, and four separate suffix model calls lose to one
vectorized eight-token prefill. The two larger cases amortize that overhead and
avoid enough projection/attention work to win. Helix therefore leaves
`CpuRuntimeConfig::prefix_cache` disabled by default; a future activation policy
needs representative workloads rather than a hard-coded crossover from this
machine. These are tiny serial CPU reference models with synthetic weights and
token IDs, not tokenizer, public-model, concurrent-batch, GPU, or production-
serving measurements.

Command:

```bash
./build-release/benchmarks/helix_benchmarks \
  '--benchmark_filter=BM_CpuRuntimeHotPrefixReuseTrace/(Disabled|Enabled)' \
  --benchmark_min_time=0.1s \
  --benchmark_repetitions=15 \
  --benchmark_enable_random_interleaving=true \
  --benchmark_report_aggregates_only=true
```

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
