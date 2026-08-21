# Helix Roadmap

## How to read this roadmap

Milestones are validation gates, not a feature wish list. A milestone is complete
only when its code, relevant tests, documentation, and required measurements exist.
The status vocabulary is defined in `AGENTS.md`.

Current environment at project creation:

- Apple Silicon host with Apple Clang 17 and CMake 4.2
- no NVIDIA GPU or CUDA compiler detected
- CPU correctness and benchmark work can be hardware-validated locally
- CUDA runtime validation remains pending access to NVIDIA hardware

No performance values are recorded until an actual benchmark is run.

## Quality gates shared by every milestone

1. Public invariants and ownership are documented.
2. Invalid inputs fail with contextual errors.
3. Deterministic correctness tests include small, odd, and boundary cases.
4. CPU-only configuration remains healthy.
5. Benchmarks separate setup from the operation being measured.
6. An optimization is called successful only after comparable before/after data.
7. README feature claims are updated to match reality.

## Phase A: foundations and correctness oracles

### M0 — Build and core foundation — Implemented and CPU-validated

Deliverables:

- CMake C++20 project and namespaced library targets
- explicit optional CUDA language/toolkit configuration
- GoogleTest and Google Benchmark integration
- compiler warnings, formatting, static analysis, sanitizer option
- CPU CI with honest CUDA exclusions
- device and dtype value types
- contextual error, assertion, and lightweight logging utilities
- initial README, license, and contribution commands

Exit criteria:

- `HELIX_ENABLE_CUDA=OFF` configures without CUDA installed
- build and CPU unit tests pass
- sanitizer build/tests pass on a supported host compiler
- format check passes
- requesting CUDA without a toolkit fails with a clear message

Validation record (2026-08-20): Apple Clang 17 on macOS arm64 configured and
built both the full and dependency-free minimal CPU configurations with warnings
as errors. All 12 discovered tests (including the benchmark smoke test) passed;
the ASan/UBSan configuration passed the same suite; and `format-check` passed.
The CUDA-on configure check produced the intended explicit missing-compiler error.
CUDA compilation and runtime execution remain unvalidated because this host has no
CUDA toolkit or NVIDIA device. Local clang-tidy execution is pending tool
availability; CPU CI installs and runs it separately.

### M1 — Tensor core — Implemented and CPU-validated

Deliverables:

- checked shape, canonical strides, numel, and byte calculations
- CPU storage with explicit alignment and RAII ownership
- contiguous Float32 tensors
- zero, deterministic random, clone/copy, safe reshape/view
- backend-neutral transfer API with CPU-to-CPU implementation
- explicit shallow-copy versus deep-copy semantics

Tests:

- scalar, zero-sized, odd-dimensional, and multidimensional tensors
- shape/byte overflow and negative dimension rejection
- reshape compatibility and shared lifetime
- copy independence and alignment
- device/dtype/access validation

Validation record (2026-08-20, commit `2887901`): contiguous Float32 tensors,
Float16 byte storage, scalar/zero-sized shapes, aligned CPU storage, deterministic
random fill, zeroing, shallow views, deep clones/copies, and CPU-to-CPU transfer
passed normal and ASan/UBSan suites. Shape, stride, and byte overflow; negative
dimensions; invalid alignment; incompatible reshape/copy; and unavailable CUDA
storage are covered. The full suite contains 29 discovered tests including the
benchmark smoke test. A release-mode `Tensor::zeros` allocation-plus-zero baseline
is recorded in `docs/PERFORMANCE.md`.

### M2 — CPU reference kernels — Implemented, CPU-validated, and benchmarked

Deliverables, in dependency order:

- add, multiply, ReLU, SiLU
- reductions over the last dimension and row-wise softmax
- matrix multiplication
- RMSNorm and SwiGLU
- CPU benchmark fixtures over representative small and inference-like shapes

Tests:

- hand-computed values and deterministic randomized references
- broadcasting is rejected until explicitly supported
- zero/one-element, odd, numerically extreme, and shape-error cases
- stable softmax and tolerance-aware comparisons

Measurements:

- release-mode latency/throughput baselines labeled as CPU reference results

Validation record (2026-08-20, commit `1488acb`): add, multiply, ReLU, stable
SiLU, full/last-dimension reduction, stable row-wise softmax, rank-two matmul,
RMSNorm, and SwiGLU passed Debug, Release, and ASan/UBSan suites. The 50 discovered
tests cover odd GEMM dimensions, zero-sized reductions/GEMM, alias rules, wide
accumulation, numerical extremes/non-finite softmax rows, shape/dtype errors, and
RMSNorm contracts. Allocation-free `_out` baselines were measured in Release mode
and are recorded in `docs/PERFORMANCE.md`. These kernels intentionally remain clear
single-threaded correctness oracles; no CPU optimization claim is made.

## Phase B: CUDA execution and kernel engineering

### M3 — CUDA runtime layer — Compile-validated; GPU runtime validation pending

- checked CUDA errors with operation context
- move-only device guard, stream, and event wrappers
- synchronous and asynchronous copy primitives
- initial direct device allocation/storage
- runtime capability reporting and hardware-gated tests

Exit criteria include leak-free resource tests on real NVIDIA hardware. Compile-only
validation is tracked separately.

Implementation/validation record (2026-08-20, commit `ebb3585`): checked CUDA
errors, device guards, move-only streams/events/buffers, synchronous/asynchronous
copies, event timing, direct CUDA Tensor storage, zeroing, cloning, and device
transfer compiled with NVCC 12.6.85/GCC 13.2 in an ARM64 CUDA 12.6 development
container. The CUDA-on warnings-as-errors build and 49 hardware-independent tests
passed. Six GPU tests compiled and skipped due the explicitly absent driver/device.
CPU Debug and ASan/UBSan regression suites remained green. Real GPU correctness,
leak, timing, and peer-copy validation is still required, so M3 is not labeled
hardware-complete. See `docs/CUDA_RUNTIME.md`.

### M4 — Basic CUDA kernels — Implemented and compile-validated; GPU runtime validation pending

- manual vector add, multiply, ReLU, and SiLU
- CPU-oracle comparisons across non-block-aligned sizes
- event-timed CPU/CUDA comparisons with warmups
- documentation of launch-overhead crossover behavior from measured data

Implementation/validation record (2026-08-20): manually implemented Float32 add,
multiply, ReLU, and numerically stable SiLU use 256-thread grid-stride kernels,
checked launch errors, explicit stream/device matching, contiguous layout contracts,
and zero-size launch suppression. Deterministic GPU tests compare against the CPU
oracles at sizes 0, 1, 31, 257, 4,099, and 65,539 and exercise in-place aliasing,
extreme SiLU inputs, and contract failures. CUDA-event benchmarks share sizes with
the CPU benchmarks and perform ten warmup launches. NVCC 12.6.85/GCC 13.2 compiled
the kernels, GPU tests, and benchmarks with warnings as errors; all 49 independent
CPU tests passed in that CUDA-on build. The GPU tests and benchmarks have not run,
so no correctness, crossover, bandwidth, or speedup result is claimed. See
`docs/CUDA_KERNELS.md`.

### M5 — Reduction and softmax — Implemented and compile-validated; GPU runtime validation pending

- naive, shared-memory, and warp-assisted reduction variants
- correct row-wise softmax followed by evidence-driven optimization
- latency, throughput, and effective-bandwidth reporting where meaningful
- documented thread layout, synchronization, and numerical behavior

Implementation/validation record (2026-08-20): last-dimension Float32 reduction
has selectable serial-per-row, shared-memory tree, and two-stage warp-shuffle
implementations. Stable row-wise softmax has selectable serial-per-row and
shared-memory block implementations, explicit NaN/infinity semantics matching the
CPU oracle, empty-row launch suppression, and safe exact in-place execution.
Deterministic GPU tests cover widths 0/1/31/257/513/1,023, multiple ranks,
large-magnitude values, non-finite rows, shape retention, algorithm validation,
aliasing, and dtype/rank errors. CUDA-event benchmarks warm up and time every
variant at 32x128, 16x1,023, and 64x4,096. A clean NVCC 12.6.85 build promoted all
CUDA warnings to errors and compiled all tests/benchmarks; 50 hardware-independent
tests passed. Static `sm_52` compiler resource data is documented, but none of the
GPU tests or benchmarks has run. No optimized-variant, bandwidth, or latency claim
is made. See `docs/CUDA_KERNELS.md`.

### M6 — Matrix multiplication — Implemented and compile-validated; GPU runtime validation pending

- one-output-element-per-thread naive SGEMM
- shared-memory tiled SGEMM supporting edge tiles
- further tiling/vectorization only after profiling
- honest cuBLAS SGEMM comparison when available
- FLOP count, traffic model, arithmetic-intensity, and occupancy analysis

Implementation/validation record (2026-08-20): rank-two row-major Float32 GEMM has
selectable one-output-per-thread and 16x16 shared-memory tiled implementations.
Both use capped two-dimensional grid-stride tile traversal, checked shape/device/
stream contracts, explicit alias rejection, asynchronous launches, zero-output
launch suppression, and correct zero-inner-dimension output. The tiled path
zero-pads every edge tile and synchronizes before consuming and replacing its two
shared tiles. GPU tests compare both variants with the double-accumulating CPU
oracle for known values, zero dimensions, and shapes including 17x19x23 and
31x33x29; contract tests cover rank, inner dimension, dtype, alias, and invalid
algorithm errors. Event fixtures cover odd, 64, 128, and 256 square/near-square
problems. NVCC 12.6.85 compiled all targets with warnings as errors; 50 independent
tests passed, while 44 GPU tests remain hardware-gated. Static `sm_52` resource
data and the traffic model are documented. GPU correctness/timing, cuBLAS
comparison, and any version-three optimization remain pending real hardware. See
`docs/CUDA_KERNELS.md`.

## Phase C: device memory systems

### M7 — Allocator core CPU-validated and benchmarked; CUDA adapter compile-validated

- aligned best-fit baseline with segment/block metadata
- split and same-segment adjacent merge
- reusable cache and explicit release policy
- reserved/allocated/free/peak/counter statistics
- specified external and internal fragmentation metrics
- native CUDA versus Helix allocation-trace benchmarks

Workloads include fixed, randomized, small-heavy, large-heavy, bursty, and
LLM-shaped traces. Strategy conclusions depend on measurements.

Implementation/validation record (2026-08-20, commit `746d0a5`): `HelixAllocator`
implements a mutex-protected, aligned best-fit segment cache over a pluggable
upstream resource. It rounds sizes, accounts for alignment prefixes, splits exact
remainders, coalesces adjacent free blocks, enforces a reservation ceiling,
releases wholly free segments explicitly, and uses allocator-instance plus
generation-tagged handles to detect cross-pool frees, double frees, and stale-
pointer ABA reuse. Stats define rounded/requested/
free/peak bytes, largest free block, active/segment/call counters, and internal/
external fragmentation. Debug builds re-derive metadata invariants after each
mutation. Thirteen allocator tests pass in normal and ASan/UBSan builds, covering
best fit, split/coalesce, alignments through 1,024 bytes on the host resource,
transactional OOM/upstream failure, failed trim, destructor ownership, and
concurrent callers. A deliberately aliased-resource test proves two pools with the
same address and local allocation ID reject each other's handles.

`CudaMemoryResource` and three hardware-gated allocator tests compile in the CUDA
12.6 warnings-as-errors build. Direct `cudaMalloc` versus Helix fixed/mixed trace
benchmarks also compile, but have not run. CPU-resource baselines were executed and
honestly show workload-dependent overhead/reuse behavior in
`docs/PERFORMANCE.md`; they are not CUDA evidence. The broader randomized,
small/large-heavy, bursty, and LLM-shaped GPU study remains pending. Immediate
reuse is deliberately not integrated into Tensor storage because stream safety is
an M8 correctness dependency. See `docs/MEMORY_ALLOCATOR.md`.

### M8 — Implemented and CPU/compile-validated; GPU runtime validation pending

- documented immediate-reuse race
- event-backed deferred free/reclamation
- explicit stream-recording contract
- real-hardware test that would detect unsafe reuse
- comparison with synchronized baseline and `cudaMallocAsync` where supported

Implementation/validation record (2026-08-20): the backend-neutral
`DeferredAllocator` retains logically freed blocks as active allocations behind a
move-only `Completion` object, opportunistically polls on allocation/trim, supports
explicit nonblocking and blocking reclamation, waits during destruction, and
reports client-versus-pending bytes/counts. Five deterministic CPU tests cover
non-reuse before completion, reuse afterward, blocking cleanup, invalid/duplicate
deferral, failure preservation, and destructor waiting. Together with M7, the full
normal and ASan/UBSan suite is 68/68.

The CUDA-only `StreamOrderedAllocator` records a timing-disabled event on a
same-device release stream and delegates lifetime to the generic state machine. A
hardware-gated test holds the stream behind a host callback, verifies the pending
address cannot be reused, releases the callback, then verifies reclamation/reuse.
NVCC 12.6.85 compiles this adapter and test with warnings as errors; all 68
hardware-independent tests pass in that build and 47 GPU tests are registered.
There is no CUDA driver/device in the container, so the event test, GPU allocation
latency, synchronized comparison, and `cudaMallocAsync` comparison remain pending.
Tensor integration also remains pending a complete multi-stream use-recording
contract; the adapter alone cannot infer outstanding work on other streams.

## Phase D: transformer primitives and memory

### M9 — CPU-validated and CUDA compile-validated; GPU runtime validation pending

- CPU and CUDA RMSNorm, RoPE, SwiGLU, and projection paths
- custom educational GEMM remains selectable
- cuBLAS production path is separately named and documented if added

Implementation/validation record (2026-08-20, commit `bd5a1ab`): CPU and CUDA
interfaces now cover RMSNorm, SwiGLU, full-dimension adjacent-pair RoPE, and
bias-free linear projection with row-major `[input_features, output_features]`
weights. RoPE consumes preselected Float32 cosine/sine rows shaped
`[tokens, head_dimension/2]`, so packed requests can provide arbitrary positions
without adding an integer Tensor dtype. The CPU path is the correctness oracle and
supports exact in-place RMSNorm/RoPE where safe.

The CUDA RMSNorm assigns a 256-thread block to each row, uses warp shuffles plus
32 bytes of shared memory for its sum-of-squares reduction, and permits exact
input/output aliasing only after the block-wide statistic is complete. SwiGLU is a
stable grid-stride pointwise kernel. RoPE assigns one thread to each adjacent pair,
loads both coordinates before writing either, and therefore supports exact
in-place rotation with pair-adjacent coalescing. Linear projection delegates to
the selectable naive/tiled M6 SGEMM; no library path is mislabeled as custom.

Four new CPU tests bring the normal and ASan/UBSan suite to 72/72. Six GPU tests
compare deterministic outputs with CPU references at empty, odd, boundary, and
non-tile-aligned shapes and validate contracts. NVCC 12.6.85 compiles all kernels,
tests, and ten-warmup CUDA-event fixtures with warnings as errors; the CUDA build
passes 72 independent tests and registers 53 GPU tests. Static `sm_52` compiler
inspection reports no spills; it is documented separately and is not runtime
evidence. CPU SwiGLU/RoPE baselines are recorded in `docs/PERFORMANCE.md`; GPU
correctness and timing await NVIDIA hardware.

### M10 — CPU-validated and CUDA compile-validated; GPU runtime validation pending

- scaled QK-transpose, causal mask, softmax, probability-times-V stages
- CPU reference and explicit temporary-memory accounting
- CUDA correctness validation before fusion work

Implementation/validation record (2026-08-20, commit `c8ac20d`): contiguous
Float32 attention defines `Q[H,Q,D]`, `K[H,K,D]`, `V[H,K,DV]`, and output
`[H,Q,DV]`. An explicit query-position offset gives one causal contract for full
prefill, chunked prefill, and single-token decode. The baseline materializes one
caller-owned `[H,Q,K]` score/probability workspace and exposes its exact
`4*H*Q*K` byte cost; `_out` calls can reuse both workspace and output.

The CPU oracle performs double-accumulating scaled QK products, writes the causal
negative-infinity mask, runs stable softmax in place, and computes the double-
accumulating probability-times-V stage. Four tests cover hand-computed causal
values, decode offset semantics, odd multi-head dimensions, zero work, resource
formulas, invalid shapes/ranges/dtypes, and alias rejection. Normal and ASan/UBSan
suites pass 76/76.

The CUDA baseline deliberately retains three visible same-stream stages: one
serial-dot-product thread per score, the M5 shared-memory softmax, and one serial-
KV-reduction thread per output value. Four CPU-oracle GPU tests and CUDA-event
fixtures compile with NVCC 12.6.85; the CUDA build passes all 76 independent tests
and registers 57 hardware-gated GPU tests. Static `sm_52` inspection reports 29
and 32 registers/thread for score and value kernels with no spills, but GPU
correctness and timing remain unvalidated. Reproducible CPU baselines and the
quadratic workspace cost are recorded in `docs/ATTENTION.md` and
`docs/PERFORMANCE.md`. No fusion or speedup claim is made.

### M11 — Implemented, CPU-validated, sanitized, and benchmarked

- layer-aware K/V shape and capacity model
- create, append, retrieve, and release sequence operations
- tests showing decode reuses prior K/V
- memory formulas checked against allocated bytes

Implementation/validation record (2026-08-21, commit `ad7ebc2`):
`ContiguousKvCache` owns per-sequence Float32 key and value tensors laid out as
`[layers, kv_heads, capacity_tokens, head_dimension]`. Whole-token append accepts
`[layers, kv_heads, head_dimension]`; layer retrieval compacts the active prefix
to `[kv_heads, length, head_dimension]`, directly matching the M10 attention
contract. The exact reservation is
`layers * 2 * kv_heads * head_dimension * capacity * sizeof(float)`, with checked
arithmetic before allocation.

Cache-instance-tagged sequence handles reject cross-cache and stale/released use.
Creation reserves full capacity transactionally under a configured byte ceiling;
full-cache and full-sequence failures leave storage, lengths, and statistics
unchanged. Release reclaims both tensors and all active capacity/token accounting.
A single documented mutex serializes mutable state. Five tests cover multi-layer
layout, cached-decode attention equivalence, exact accounting, invalid contracts,
transactional exhaustion, double release, and concurrent appends. Normal and
ASan/UBSan suites pass 81/81. The CUDA-on build also passes all 81 independent
tests; 57 GPU tests remain hardware-gated, and M11 does not claim a GPU cache path.
CPU lifecycle and allocation-free layer-copy baselines are recorded in
`docs/PERFORMANCE.md`; design and limitations are in `docs/KV_CACHE.md`.

### M12 — Implemented, CPU-validated, sanitized, and benchmarked

- fixed-token physical blocks and strongly typed block/sequence IDs
- logical-to-physical block tables and free-block pool
- transactional append on block exhaustion
- deterministic allocation/reclamation tests
- fragmentation/capacity comparison with contiguous sequence allocation

Reference counting is deferred until prefix sharing creates a real need.

Implementation/validation record (2026-08-21, commit `68e8d31`):
`PagedKvCache` owns a fixed Float32 pool laid out as
`[physical_blocks, layers, kv_heads, block_tokens, head_dimension]` for both keys
and values. Strong `PhysicalBlockId` and cache-tagged `PagedSequenceHandle` types
separate physical ownership from logical sequence identity. Per-sequence block
tables translate logical token positions; a preallocated circular FIFO provides
deterministic allocation and allocation-free reclamation.

An append that crosses a block boundary grows the sequence table before dequeuing
a block. Input failure, metadata OOM, or an empty free pool therefore leaves the
mapping, length, payload, and success counters unchanged. A partially filled final
block remains writable with zero blocks free. Release prevalidates ownership and
returns every block exactly once. Debug builds re-derive table lengths, owners,
free membership, token slots, bytes, and counters after mutations.

Seven new tests bring normal and ASan/UBSan suites to 88/88. They cover values
across block boundaries, attention interoperability, transactional exhaustion,
FIFO reuse, stale-data invisibility, invalid and overflowing contracts, exact
reclamation/accounting, a contiguous tail-slack comparison, and concurrent
callers. The CUDA-on build passes all 88 hardware-independent tests; the existing
57 GPU tests remain hardware-gated. M12 is a CPU metadata/storage implementation,
not a GPU-runtime claim.

Release measurements retain both sides of the tradeoff. With a long-lived pool,
paged lifecycle traces were 9.5%, 72.0%, and 14.9% lower CPU time than the
corresponding contiguous traces, but their allocation boundaries and reserved
capacities differ as documented. Compact layer copying was 2.35x and 1.75x slower
for the 128 KiB and 2 MiB histories because it traverses 8 and 32 blocks; the odd
25,752-byte case was within run-to-run variation. See `docs/KV_CACHE.md` and
`docs/PERFORMANCE.md` for exact methodology and limitations.

### M13 — Paged attention — CPU-validated, sanitized, and benchmarked; GPU path pending

- correctness path reading K/V through sequence block tables
- comparison with contiguous attention
- memory-access and coalescing analysis
- measured latency, overhead, fragmentation, and capacity on target hardware

Implementation/validation record (2026-08-21, commits `26a9adf`, `2977005`, and
`aaf9d59`): `PagedKvCache::LayerReadView` pins cache storage and sequence metadata
behind a move-only read lease, exposes only written per-block prefixes, and holds
the cache mutex for its lifetime so append/release cannot invalidate a traversal.
The direct Float32 CPU attention path consumes those block spans without creating
a compact K/V history. It retains the M10 `[H,Q,K]` score/probability workspace,
causal offset semantics, stable softmax, and double accumulation.

Four paged-attention tests compare full prefill and chunked decode with contiguous
attention across odd dimensions and block boundaries, cover zero queries, and
exercise invalid shape/dtype/range/output/alias contracts. A separate cache test
proves the lease pins storage after its cache facade is destroyed and exposes only
the final block's written prefix. Normal and ASan/UBSan suites pass 93/93; the
CUDA-enabled build passes the same 93 hardware-independent tests and still
registers 57 hardware-gated GPU tests.

The first value stage walked every page once per output coordinate. Measurement
identified that repeated page lookup and strided value access as the bottleneck.
A fixed 32-coordinate stack tile now walks each page once per tile while preserving
each coordinate's KV accumulation order and requiring no dynamic allocation. The
original dimension-major path remains callable for paired benchmarking. Applying
the same tile to contiguous attention avoids crediting a general loop-order
improvement to paging. In a same-process Release run, tiling reduced paged CPU
median time by 2.36x–2.99x; with equal tiled traversal, direct paging cost 2.1%–
6.7% more CPU time than contiguous storage across the four measured cases. Exact
shapes, variation, methodology, and caveats are in `docs/PERFORMANCE.md`.

This completes the CPU correctness and measurement gate, not the GPU portion of
M13. There is no device-resident block pool/table or CUDA paged-attention kernel,
so coalescing, occupancy, GPU latency, and capacity remain pending NVIDIA hardware
and implementation.

## Phase E: requests and continuous execution

### M14 — Request model — Implemented, CPU-validated, and sanitized

- stable request/token IDs
- validated waiting, prefill, decoding, terminal state transitions
- generation limits, cancellation, failure context, and timestamps
- exhaustive state-transition tests

Implementation/validation record (2026-08-21, commit `582357b`): strong nonzero
64-bit `RequestId` and independent 32-bit `TokenId` types feed a move-only,
single-owner `Request`. The explicit waiting/prefill/decoding/finished/cancelled/
failed graph has no terminal exits. Positive chunked prefill progress transitions
to decode only after the whole prompt, or finishes immediately when
`max_new_tokens` is zero. Generated tokens retain emission times and finish
atomically at the configured length; EOS, cancellation, and failure have separate
reasons, with context retained for the latter two.

All event times use `steady_clock` and must be monotonic. Created, first-scheduled,
prefill-complete, first/last-token, and completion times support derived waiting
and TTFT metrics without a nondeterministic internal clock call during mutation.
Ten tests enumerate all 36 state pairs and cover chunking, limits, EOS, every
active-state cancellation/failure path, invalid inputs, time regression,
transactional failure, enum validation, metrics, and move ownership. Normal and
ASan/UBSan suites pass 103/103. The CUDA-enabled build passes the same 103
hardware-independent tests; 57 GPU tests remain gated. See `docs/SCHEDULER.md`.

### M15 — FIFO/token-budget scheduler — Planned

- waiting/running ownership and deterministic FIFO admission
- iteration token budget and KV-block capacity interface
- separate prefill/decode work descriptions
- completion, cancellation, and resource-exhaustion behavior

### M16 — Continuous batching — Planned

- removal/admission between inference iterations
- deterministic mixed-length workload tests
- synthetic static-versus-continuous benchmark
- requests/sec, tokens/sec, wait, TTFT, inter-token, p50, and p95 definitions

Metrics are published only for measured workloads and include the workload
distribution and concurrency.

## Phase F: narrow end-to-end inference

### M17 — Minimal transformer runtime — Planned

- one precisely documented Llama-style architecture
- embedding, pre-norm blocks, Q/K/V, RoPE, attention, projection, SwiGLU,
  residuals, final norm, and LM head
- versioned checked Helix weight format
- one-time model conversion tool if required
- tiny deterministic model compared layer-by-layer with an independent oracle

### M18 — Autoregressive generation — Planned

- prefill, cached decode, greedy token selection, and stop conditions
- supported tokenizer boundary documented without making Python a runtime dependency
- minimal `helix generate` CLI and reproducible example

### M19 — Runtime execution engine — Planned

- single owner connecting scheduler, model executor, KV cache, allocator, and stream
- request submission/result API
- lifecycle metrics and resource cleanup under success/cancel/failure
- end-to-end deterministic and stress tests

## Phase G: measured advanced work

These items remain optional until the single-device runtime is correct and
profiled.

### M20 — Stream overlap — Optional

Pinned memory, multiple streams, and event dependencies; retain only overlap that
improves a measured workload.

### M21 — Kernel fusion — Optional

Profile-driven candidates such as bias/activation, RMSNorm, or SwiGLU. Record
latency, launch count, and traffic before and after.

### M22 — Prefix cache — Optional

Content key, KV block reference counts, ownership, eviction, and concurrency tests.

### M23 — Memory pressure and preemption — Planned baseline, optional extension

The first policy is safe admission rejection. Preemption or prefix eviction is an
optional measured extension with explicit fairness and restoration semantics.

### M24 — Additional precision/quantization — Optional

FP16, BF16, then INT8 only when hardware support, numerical contracts, kernels,
and representative accuracy/performance tests exist.

### M25 — Multi-GPU tensor parallelism — Optional; multi-GPU hardware required

Start with a sharded linear-layer demonstration and NCCL collectives. Never label
mock or single-GPU tests as distributed validation.

## Documentation deliverables by subsystem

- `docs/MEMORY_ALLOCATOR.md` begins with M7 and gains stream safety in M8.
- `docs/CUDA_KERNELS.md` begins with M4 and records each measured kernel evolution.
- `docs/KV_CACHE.md` begins with M11 and adds paging/attention in M12–M13.
- `docs/SCHEDULER.md` begins with M14 and adds continuous batching in M16.
- `docs/PERFORMANCE.md` starts with the first benchmark and never contains
  illustrative numbers in result columns.
- `docs/MODEL_SUPPORT.md` is created before M17 compatibility claims.
- `docs/INTERVIEW_GUIDE.md` evolves with implemented code and cites actual types.
- `docs/RESUME.md` contains only validated implementation and measured claims.

## Strong portfolio release gate

The first portfolio release requires excellent M0–M8 work, correct attention and
paged KV state through M13, deterministic continuous batching through M16, and a
credible attempt at M17–M19 without weakening earlier quality. If end-to-end model
execution is deferred, the release must say so plainly and present the deeply
validated lower-level systems as the project scope.
