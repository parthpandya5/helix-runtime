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
- contiguous Float32 and UInt32 token-index tensors
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

Token-index follow-up (2026-08-24, commit `61b3768`): UInt32 now has stable dtype
metadata, typed Tensor construction/access, zeroing, views, clones, and CPU/CUDA
transfer through the existing byte-exact storage paths. Float16 remains a storage-
only type. Debug, Release, and ASan/UBSan suites pass 191/191; the CUDA compile
suite passes 190 host tests and skips 69 hardware-gated tests (259 total).

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
12.6 warnings-as-errors build. Direct `cudaMalloc` versus Helix fixed, random-size,
small-heavy, large-heavy, bursty, and LLM-like trace benchmarks also compile, but
have not run. CPU and CUDA paths reuse exact deterministic trace definitions.
CPU-resource baselines were executed and honestly show workload-dependent overhead/
reuse behavior in `docs/PERFORMANCE.md`; they are not CUDA evidence. The complete
GPU study remains pending. Immediate CUDA reuse is deliberately not integrated into
Tensor storage because stream safety is an M8 correctness dependency. See
`docs/MEMORY_ALLOCATOR.md`.

CPU Tensor-ownership follow-up (2026-08-24, commit `4f6fc7f`): mutable external
Storage now validates pointer/capacity/alignment/owner invariants, and the opt-in
`allocate_cpu_tensor` bridge retains both a shared allocator and exact allocation
descriptor until the final Tensor/view releases Storage. Logical and rounded bytes
remain distinct; zero work bypasses the pool; invalid shape/alignment/device/OOM
requests create no active handle; and CUDA resources are rejected before upstream
allocation. Three bridge tests plus three mutable Storage/Tensor tests cover reuse,
view and allocator lifetime, overflow, capacity, alignment, and failure
transactionality. Debug, Release, and ASan/UBSan suites pass 184/184; a CUDA 12.6
warnings-as-errors build passes 183 hardware-independent tests and skips 57 GPU
tests (240 total) without a driver/device.

A paired pre-warmed host Tensor lifecycle benchmark retains the direct factory.
The bridge was about 2.04x slower for one and 1,024 Float32 elements, but its 239 ns
median was 47.9% below the 459 ns direct median for 1,048,576 elements (4 MiB
logical). The default pool retained 64 MiB. This is an opt-in CPU ownership result,
not evidence about `cudaMalloc`; full methodology is in `docs/PERFORMANCE.md`.

Allocator workload/metadata follow-up (2026-08-24, commit `32184e7`): five
self-validating traces cover randomized sizes, small-heavy and large-heavy churn,
bursts, and an LLM-like mix of persistent weights, rolling KV pages, and temporary
workspaces. A separate untimed replay samples peak rounded bytes and maximum
internal/external fragmentation after every operation. The original pool rebuilt
block vectors and churned hash nodes; capacity-planned in-place split/coalesce and
a generation-checked reusable active-slot registry reduced median cached-trace CPU
time by 53.1%–66.7%. Final host results retain both outcomes: random/large/LLM-like
traces were 3.13x–9.41x faster than direct aligned allocation, while small-heavy/
bursty remained 2.34x–2.45x slower. Debug, Release, and ASan/UBSan suites pass
186/186; the CUDA-on build passes 185 host tests and explicitly skips all 57 GPU
tests (242 total) without a driver/device.

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
There is no CUDA driver/device in the container, so the event test and all GPU
allocation timing remain pending.
General CUDA Tensor integration remains pending a complete multi-stream use-
recording contract; the event adapter alone cannot infer outstanding work on other
streams.

Single-stream Tensor follow-up (2026-08-24):
`allocate_cuda_tensor_on_stream` now wraps a cached device allocation in shared
Tensor Storage, retains its allocator and one usage stream through the final view,
then records the release event automatically. The allocation stays pending and
unavailable for reuse until that event completes. Constructor rollback returns an
unused block immediately; a non-throwing destructor falls back to stream
synchronization if event deferral fails. This is an explicit one-stream contract,
not a claim of arbitrary record-stream tracking.

Three new hardware-gated tests cover final-view lifetime, a host-gated stream that
prevents premature address reuse, post-event reuse, allocator/stream ownership,
zero work, shape/size/alignment/owner failures, and moved-from streams. During the
same audit, pending lookup changed from local numeric ID comparison to exact
allocator/generation-tagged handle identity, eliminating cross-pool ID collisions.
Direct and pooled Tensor lifecycle benchmarks perform ten warmups and compare
complete host lifecycle cost at 1, 1,024, and 1,048,576 Float32 elements. NVCC
12.6.85 compiles all paths with warnings as errors; the CUDA suite passes 185 host
tests and registers 66 GPU tests (251 total), all skipped without a driver/device.
No lifecycle latency or event-correctness result is claimed.

CUDA Toolkit 11.2+ builds now also register a default-pool
`cudaMallocAsync/cudaFreeAsync` comparison. It checks device memory-pool support,
performs ten warmups, preserves and reports the pool release threshold, and
synchronizes each timed allocation/free pair. The fixture compiles with CUDA
12.6.85; it has not executed without a device, so it closes the benchmark-harness
gap but not the M8 measurement gate.

## Phase D: transformer primitives and memory

### M9 — CPU-validated and CUDA compile-validated; GPU runtime validation pending

- CPU and CUDA embedding, token/head layout, RMSNorm, RoPE, SwiGLU, and projection
  paths
- custom educational GEMM remains selectable
- cuBLAS production path is separately named and documented if added

Implementation/validation record (2026-08-20, commit `bd5a1ab`): CPU and CUDA
interfaces now cover RMSNorm, SwiGLU, full-dimension adjacent-pair RoPE, and
bias-free linear projection with row-major `[input_features, output_features]`
weights. RoPE consumes preselected Float32 cosine/sine rows shaped
`[tokens, head_dimension/2]`, so packed requests can provide arbitrary positions
without coupling RoPE itself to an integer position tensor. The CPU path is the
correctness oracle and supports exact in-place RMSNorm/RoPE where safe.

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

Embedding follow-up (2026-08-24, commit `61b3768`): the model's former private row
copy is now a reusable UInt32 `[tokens]` plus Float32 `[vocabulary,width]` kernel.
CPU lookup prevalidates all IDs before any output write and rejects full or partial
logical-range aliasing. The asynchronous CUDA baseline assigns one block per token,
strides 256 threads across row width, and guards each device ID before table address
formation; invalid IDs produce a NaN row rather than an out-of-bounds read. Host
metadata, device, stream, shape, dtype, and range-overlap contracts are checked
before launch.

Three CPU tests cover repeated/out-of-order rows, empty and odd shapes, late OOV
failure atomicity, dtype/shape errors, and partially overlapping external buffers.
Three hardware-gated GPU tests add exact CPU-oracle comparisons through 257x513,
invalid-ID sentinels, moved streams, and partial CUDA-buffer overlap. The Release
CPU baseline performs ten explicit warmups; the CUDA fixture performs ten launch
warmups, validates against the CPU oracle before and after timing, and uses same-
stream CUDA events. CPU measurements are recorded in `docs/PERFORMANCE.md`. NVCC
12.6.85/GCC 13.2 compile all paths with warnings as errors; GPU execution and
timing remain pending.

Head-layout follow-up (2026-08-24, commit `8e1c582`): the model-private Q/K/V and
attention-context reorder loops are now checked shared kernels. Both directions
swap the first two axes of contiguous rank-three Float32 storage, preserve the
innermost head vector, return exact swapped empty shapes, and reject exact or
partial logical-range overlap. `CpuLlamaModel` uses the shared operations for full
prefill and both contiguous/paged decode paths; independent model, generation, and
runtime tests remain exact within their existing tolerances.

The asynchronous CUDA baseline assigns one 256-thread block to each head vector,
caps and grid-strides the vector count, and strides threads across head dimension.
Two hardware-gated tests compare both directions exactly with the CPU oracle over
empty, tiny, odd, and non-block-aligned shapes, then validate rank, dtype, shape,
device, moved-stream, exact-alias, and partial-buffer-overlap failures. Matching
CPU/CUDA benchmarks cover decode, prefill, odd, and nonstandard geometry with ten
explicit warmups; CUDA uses pre/post oracle checks and same-stream events. Static
`sm_52` inspection reports 18 registers/thread and no shared/local memory or
spills, which is compile evidence only. Measured CPU baselines are in
`docs/PERFORMANCE.md`; no speedup over the unbenchmarked private loops is claimed.
Debug, Release, and ASan/UBSan suites pass 195/195. The CUDA-on build passes 194
host tests and skips 71 GPU tests without a driver/device (265 total).

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

M12 deliberately deferred reference counting until prefix sharing created a real
need; the M22 record below adds it without changing the original paging baseline.

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

### M13 — Paged attention — CPU measured; CUDA baseline compile-validated

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

Post-M19 integration (2026-08-23, commit `c9b151c`) adds a cached-history-plus-
tail attention contract. The model can traverse paged spans while treating current
K/V as an uncommitted causal tail, so token execution no longer needs two compact
history-plus-tail tensors per layer. The compact path remains selectable. Two new
attention tests cover odd multi-token tails, empty history, causal masking, and
invalid/alias contracts; model and runtime tests compare both decode modes.
Isolated paired measurements reduced median CPU time by 13.3%–27.7% and removed
2 KiB–1 MiB of temporary combined K/V, while full tiny runtime traces improved by
only 0.08%–6.23%.

CUDA baseline record (2026-08-24): an immutable move-only device block table and
Float32 page kernels now consume layer-local
`[physical_blocks,heads,block_tokens,width]` K/V tensors without compaction. The
three explicit stages remain paged score/causal mask, shared stable softmax, and
paged probability-times-V. Four GPU tests compare reordered/non-divisible pages
with the contiguous CPU oracle, poison unwritten tail slots, cover odd dimensions
and empty work, and exercise invalid table/shape/offset/device/alias contracts.
Four CUDA-event fixtures pair the existing logical attention sizes with reversed
page mappings and account for active bytes, physical capacity, table bytes, and
workspace.

NVCC 12.6.85 compiled the implementation, tests, and benchmarks with line
information and warnings as errors. The CUDA-on suite passes 185 host tests and
registers 66 GPU tests (251 total), all correctly skipped without a driver/device.
Static `sm_52` inspection reports 31/29 registers per thread for the paged
score/value kernels with no shared/local memory or spills. These are compiler facts,
not execution evidence. Runtime correctness, event timing, coalescing/occupancy
profiling, a GPU-resident KV pool, and allocator/executor integration remain gated
on NVIDIA hardware and later work.

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

### M15 — FIFO/token-budget scheduler — Implemented, CPU-validated, sanitized, and benchmarked

- waiting/running ownership and deterministic FIFO admission
- iteration token budget and KV-block capacity interface
- separate prefill/decode work descriptions
- completion, cancellation, and resource-exhaustion behavior

Implementation/validation record (2026-08-21, commit `7b50dbc`): a single-thread-
owned scheduler retains each move-only request, maintains deterministic waiting and
running order, and emits typed prefill ranges or one-token decode work. Scheduling
is bounded independently by tokens, resident requests, and a caller-provided KV
free-block snapshot. Each work record exposes blocks before/new; completion,
failure, and cancellation report exact logical release counts without pretending
the scheduler owns physical KV memory. Partial-block slack remains usable with no
free blocks, while an unschedulable FIFO head yields a partial/empty result rather
than being skipped or failed.

Only one iteration may be in flight. Completion validates the full ordered result
vector before mutation, then stages failure strings and token-vector capacity so
logical request progress cannot be split by an allocation failure. Ten scheduler
tests cover budgets, chunking, resource exhaustion, state/result mismatches,
whole-vector transactionality, cancellation/failure, monotonic time, ownership,
and statistics. At the M15 gate, normal, ASan/UBSan, and CUDA-enabled host suites
passed 113/113; all CUDA sources compiled with NVCC 12.6.85, while 57 GPU tests
remained hardware-gated.

A four-case Release trace records complete host policy cost (48.777 us to 1.342 ms
median CPU time) but deliberately excludes model math and physical KV operations;
it is not an inference-throughput claim. See `docs/SCHEDULER.md` and
`docs/PERFORMANCE.md`.

### M16 — Continuous batching — Implemented, CPU-validated, sanitized, and benchmarked

- removal/admission between inference iterations
- deterministic mixed-length workload tests
- synthetic static-versus-continuous benchmark
- requests/sec, tokens/sec, wait, TTFT, inter-token, p50, and p95 definitions

Implementation/validation record (2026-08-21, commit `f17ce74`): completion removes
terminal requests from the resident order, and the next iteration fills each freed
slot from the waiting FIFO while the surviving cohort continues. An exact test
executes A/B/C/D, finishes A, then observes B/C/D plus newly admitted E in the next
mixed decode/prefill batch.

`SchedulerMetrics` uses an explicit observation boundary and reports lifecycle
state/token counters, successful requests/second, generated and processed
tokens/second, and sample count/min/mean/p50/p95/max for waiting, TTFT, successful
request latency, and adjacent-token intervals. Percentiles use documented linear
interpolation; zero windows do not fabricate rates; invalid IDs, duplicate samples,
and time regression are rejected. Three focused metrics tests cover exact values,
active/terminal states, empty samples, and invalid snapshots.

The static-versus-continuous benchmark uses two disclosed mixed-length workloads,
sufficient synthetic KV capacity, and a deliberately constant 1 ms modeled cost
per iteration. The model produced 48/132 and 92/260 continuous/static iterations,
or 2.75x and 2.83x modeled request/token rates. These are policy-model results, not
measured inference throughput. Measured host control-path CPU times were close and
do not support an optimization claim. Normal, ASan/UBSan, and CUDA-enabled host
suites pass 118/118; 57 GPU runtime tests remain gated. See
`docs/PERFORMANCE.md` for workloads, exact distributions, host timings, and the
benchmark-observer correction.

## Phase F: narrow end-to-end inference

### M17 — Minimal transformer runtime — Implemented, CPU-validated, sanitized, and benchmarked

- one precisely documented Llama-style architecture
- embedding, pre-norm blocks, Q/K/V, RoPE, attention, projection, SwiGLU,
  residuals, final norm, and LM head
- versioned checked Helix weight format
- one-time model conversion tool if required
- tiny deterministic model compared layer-by-layer with an independent oracle

Implementation/validation record (2026-08-23, commit `987c3b2`):
`CpuLlamaModel` implements full causal prefill for a deliberately narrow Float32
Llama-style architecture: equal Q/K/V head counts, even full rotary head
dimensions, bias-free row-major projections, pre-attention/pre-MLP RMSNorm,
adjacent-pair RoPE, causal attention, residuals, SwiGLU, final norm, and an untied
LM head. The ordinary path returns `[tokens,vocabulary]` logits; an opt-in trace
retains every post-layer hidden state and final normalized state for diagnostics.

A standalone vector oracle independently evaluates all stages for a deterministic
two-layer, odd-length model. Tests compare each layer state, the final normalized
state, and every logit, then cover maximum/empty/overlength/OOV inputs, invalid
configuration geometry, dtype/layer-count/shape errors, and repeat determinism.

The `HELIXLM1` format uses a fixed little-endian header and directory, exact tensor
roles/shapes, 64-byte payload alignment, checked offsets/lengths, zero padding,
metadata and per-payload FNV-1a checksums, finite-weight validation, and exact file
coverage. Tests exercise bit-exact deterministic round trips plus truncated,
trailing, corrupted, noncanonical, non-finite, and structurally invalid inputs.
FNV detects accidental corruption and is explicitly not an authenticity scheme.

Normal, Release, and ASan/UBSan CPU suites pass 129/129. The CUDA 12.6.85/GCC 13.2
warnings-as-errors build compiles all targets and passes the same 129 host tests;
57 GPU tests remain hardware-gated. Release CPU prefill and checked in-memory load
baselines are recorded in `docs/PERFORMANCE.md`. This completes M17's narrow CPU
gate, not CUDA model execution or general checkpoint compatibility. A public
checkpoint converter was not added because no checkpoint is yet claimed; the
native writer API is tested and ready for a later explicit mapping tool. See
`docs/MODEL_SUPPORT.md`.

Conversion follow-up (2026-08-24, commit `4c13d59`): the later explicit mapping
tool is now implemented for one documented Hugging Face Llama safetensors layout.
The Python/NumPy stage strictly parses single or indexed/sharded F32/F16/BF16
payloads, validates the narrow architecture, transposes PyTorch linear weights,
and interleaves per-head Q/K coordinates to reconcile half-split and adjacent-pair
RoPE. A streaming C++ raw-payload writer remains the `HELIXLM1` format authority;
it validates count, exact sizes, finiteness, input/output aliasing, and checksums
without retaining a second complete weight bundle.

The synthetic integration fixture covers mixed precision, sharding, atomic output
replacement, tied-embedding fallback, path confinement, rejected GQA/shape/non-
finite inputs, exact container contents, and C++ execution. Its independent
source-layout oracle compares Hugging Face-style half-split-RoPE logits with the
converted Helix model. Debug, Release, and ASan/UBSan CPU suites pass 172/172.
The CUDA 12.6 compile container builds both C++ tools, passes 171 host tests, and
explicitly skips 57 GPU tests (228 total); its absent Python/NumPy means the one
converter integration test is not registered there. At that checkpoint, neither
a downloaded public model nor a tokenizer had been validated, so broad
compatibility was not claimed. Details are in `docs/MODEL_SUPPORT.md`.

Immutable-loading follow-up (2026-08-24, commit `fc7fff5`): Tensor/Storage now
supports aligned externally owned CPU ranges with an enforced read-only contract.
The opt-in `read_llama_model_file_mapped` path uses a private POSIX mapping on
little-endian IEEE-754 hosts, shares one RAII mapping lifetime across payload-sized
Tensor views, and retains the copied stream/filesystem loader as the portable
baseline. Both paths use the same metadata parser and verify exact coverage,
padding, payload checksums, finite values, and final weight geometry before return.

Tests prove bit-exact copied equivalence, mutation rejection, clone/view semantics,
64-byte alignment, executable logits, mapping survival after unlink, and rejection
of truncated, trailing, corrupted, non-finite, and missing inputs. A randomized
paired warm-page benchmark measured 20.96%, 15.14%, and 12.78% lower median CPU
load time across the three disclosed 117,184 B–3,153,600 B synthetic files. It is
not a cold-disk, RSS, inference, or GPU result. Debug, Release, and ASan/UBSan
native suites pass 184/184. The CUDA 12.6 compile container passes 183 hardware-
independent tests and explicitly skips 57 GPU tests without a device (240 total);
its separate converter integration test remains unavailable because Python/NumPy
is absent.

Public-checkpoint validation follow-up (2026-08-24, commit `d433760`): a bounded
offline validator independently parses source-layout safetensors, evaluates the
supported Llama equations including half-split RoPE, and compares the complete
last-token logit vector emitted by the C++ inspector. Its default five-million-
parameter ceiling keeps the explicit reference loops scoped to small correctness
models. Synthetic integration now covers success, vocabulary rejection, budget
refusal, and a deliberately mismatched source/model pair in addition to the
converter checks above.

The revision-pinned public artifact
`llamafactory/tiny-random-Llama-3@c17693c90450da4160582668bfe65a60175d43e5`
converted to a deterministic 1,032,272-parameter Float32 `HELIXLM1` file. Four
raw-token sequences spanning one through seven positions and token IDs 0 through
31,999 produced 128,000 logits; every element passed `rtol=2e-5`, `atol=2e-6`
against both the independent NumPy source oracle and optional CPU Float32 PyTorch
2.0.1 / Transformers 4.32.0 execution. Maximum absolute errors were `2.98e-8`
and `8.94e-8`, respectively. Exact input/output hashes are recorded in
`docs/MODEL_SUPPORT.md`. This is one random checkpoint and no tokenizer, broad
compatibility, learned language quality, or CUDA model result is claimed.

### M18 — Autoregressive generation — Implemented, CPU-validated, sanitized, and benchmarked

- prefill, cached decode, greedy token selection, and stop conditions
- supported tokenizer boundary documented without making Python a runtime dependency
- minimal `helix_generate` token-ID CLI and reproducible example

Implementation/validation record (2026-08-23, commit `c515d6d`):
`CpuLlamaGenerator` owns one validated CPU model, one fixed-capacity
`ContiguousKvCache`, and one cache-tagged sequence. Full prefill captures each
layer's rotated head-major keys and unrotated values and appends prompt positions
only after model execution succeeds. One-token decode requires the cache length to
equal the RoPE position, validates model/cache geometry, reads the prior prefix,
computes the complete layer stack, and returns all layers' new K/V separately so
the session mutates only after successful execution.

Greedy selection reads the final logits row, rejects non-finite values, and uses
the lowest token ID for equal maxima. Generation reports distinct maximum-token,
EOS, and context-capacity stops. The last emitted token is not cached unless it is
fed through a subsequent decode step; the returned cache count makes that
contract visible. The minimal `helix_generate` executable reads a checked
`HELIXLM1` file and accepts token IDs directly. No tokenizer or public-checkpoint
compatibility is implied by the generation CLI itself; the later narrow converter
does not change this token-ID boundary.

Deterministic tests compare prefill and successive cached-decode logits with full
prefix recomputation, compare complete greedy token sequences, cover position
zero, tie-breaking, non-finite logits, EOS/zero-budget/context stops, geometry and
state errors, and verify failures do not advance cache state. Decode history was
first implemented with an owning compact snapshot followed by a second copy into
the final attention buffer. A measured direct-prefix API removes two temporary
tensors and one history copy per layer while retaining cache locking and stale-
length validation.

Normal, Release, and ASan/UBSan CPU suites pass 136/136. A CUDA 12.6.85/GCC 13.2
warnings-as-errors build compiles every target and passes the same 136 host tests;
all 57 registered GPU tests remain skipped without an NVIDIA driver/device. On the
disclosed Apple M3 Pro CPU fixtures, cached next-position decode measured 3.69x,
23.59x, and 56.79x faster than full-prefix recomputation at total sequence lengths
8, 32, and 64. The direct-prefix staging change reduced its isolated median by
45.7%, 45.4%, and 58.1%. Exact shapes, timing boundaries, variation, and commands
are in `docs/PERFORMANCE.md`; these are tiny single-threaded CPU reference results,
not production or GPU inference claims.

Public-generation validation follow-up (2026-08-24, commits `f2f8ea5` and
`09fd7aa`): the C++
inspector exposes the existing contiguous-KV session as a machine-readable greedy
trace. The bounded Python harness compares that cached path against repeated full-
prefix evaluation in the independent source layout and can additionally compare
with Transformers `past_key_values`. Synthetic integration covers normal decode,
invalid length, and context exhaustion; existing C++ tests retain EOS, tie,
failure-atomicity, and per-step logit coverage. On the revision-pinned public
checkpoint, prompt `1,42,7` produced the identical eight-token trace
`9555,23124,30954,4355,14017,25190,6074,6566` through all three paths, with the
same maximum-token stop and cached count of 10. This is raw-token parity for one
random model, not tokenizer, text-generation quality, intermediate-logit parity
at every public decode step, or CUDA execution.

### M19 — CPU runtime execution engine — Implemented, CPU-validated, sanitized, and benchmarked

- single owner connecting scheduler, CPU model executor, and paged KV cache;
  CUDA allocator/stream ownership remains future work
- request submission/result API
- lifecycle metrics and resource cleanup under success/cancel/failure
- end-to-end deterministic and stress tests

Implementation/validation record (2026-08-23, commits `8086257` and `2a6b365`):
`CpuRuntimeEngine` is a
single-thread-owned synchronous orchestrator over one `FifoScheduler`, one
`CpuLlamaModel`, one global `PagedKvCache`, and a request/session map. Each call
uses the cache's actual free-block snapshot, executes the scheduler's ordered mixed
prefill/decode work, completes logical state, and reclaims every terminal physical
sequence before returning. Submission validates vocabulary, context, unique IDs,
and whether the maximum live K/V prefix can fit in the pool; partial insertion
rolls back its empty sequence.

`CpuLlamaModel::decode_one` now accepts either contiguous or paged history. A
transactional paged bulk-append API consumes full-prefill layer-major artifacts
across partial/whole pages. The default engine uses one full causal prefill when a
complete new prompt is scheduled and retains token-at-a-time prefill for arbitrary
chunks and paired measurement. Scheduler capacity accounting now recognizes that
the final length-limited output is selected from existing logits and needs no new
K/V, preventing an exact-block-boundary deadlock.

Nine engine tests cover full and chunked prefill, greedy equivalence with full
recomputation, genuine mixed-work continuous replacement, physical exhaustion and
reuse, EOS/cancellation, invalid submission rollback, injected model failure, and
complete reclamation. One compares compact and direct paged decode through complete
requests. The stress case runs 32 mixed-length requests through a constrained
pool, compares every output with full recomputation, and verifies reuse plus final
zero ownership. Normal, Release, and ASan/UBSan suites pass 152/152. A CUDA 12.6.85
warnings-as-errors build compiles every target and passes the same 152 host tests;
57 GPU tests remain skipped without a driver/device.

On three disclosed tiny Apple M3 Pro CPU traces, the full-prompt default reduced
median process CPU time by 51.3%, 36.0%, and 15.8% versus retained sequential-token
prefill with compact-copy decode pinned in both cases. Exact model/request shapes,
peak K/V bytes, iterations, timing boundaries, variation, and command are in
`docs/PERFORMANCE.md`. This completes M19's CPU correctness-engine gate, not CUDA
model execution: work items are serial, the pool is host Tensor storage, and
streams/events plus `HelixAllocator` integration remain pending. See
`docs/RUNTIME.md`.

The direct-decode integration in commit `c9b151c` makes block-table plus
uncommitted-tail attention the measured default while retaining compact copy in
`CpuRuntimeConfig`. See the M13 post-integration record and paired measurements in
`docs/PERFORMANCE.md`; no GPU result is implied.

## Phase G: measured advanced work

These items remain optional until the single-device runtime is correct and
profiled.

### Profiling foundation — Implemented and contract/compile-validated

`scripts/profile_cuda.sh` selects one exact CUDA benchmark and emits bounded
Nsight Systems, Nsight Compute, or combined commands. Actual collection requires a
visible device and records the exact command, benchmark, launch bounds, Git/build
state, GPU/driver/toolkit/profiler versions, OS, CPU, and RAM next to ignored
reports. Existing/symlink outputs and unsafe stems are rejected; `--force` is
explicit. A CPU CTest verifies dry-run purity, bounds, failure cases, and force
propagation. `HELIX_ENABLE_CUDA_LINEINFO` adds profiler source correlation only to
CUDA compilation. The option and all CUDA targets compile with CUDA 12.6.85 and
warnings as errors; no report was collected without NVIDIA hardware. Full usage
and the rule excluding profiler-perturbed timings from benchmark claims are in
`docs/PERFORMANCE.md`.

### M20 — Transfer/overlap experiment compile-validated; measurement pending

`cuda::PinnedHostBuffer` now provides move-only RAII around portable page-locked
host memory, including zero-size and non-throwing cleanup semantics. One
hardware-gated test moves ownership and performs an odd 4,099-byte H2D/D2H round
trip on one nonblocking stream. Pageable-versus-pinned H2D and D2H fixtures cover
4 KiB, 1 MiB, and 16 MiB, perform ten warmups, and use same-stream CUDA events for
transfer duration. Host allocation and initialization are outside timing.

The paired overlap fixture retains one exact eight-chunk pinned H2D → custom SiLU
→ pinned D2H workload. Its baseline submits every operation to one stream. Its
double-buffered variant alternates two device slots and two transfer streams, uses
a serialized compute stream, and connects input-ready, compute-ready, slot-done,
and common timing boundaries with CUDA events. A five-chunk, 257-element odd-size
test compares the complete pipelined output with the CPU SiLU oracle. Benchmark
cases use 16,384, 262,144, and 1,048,576 elements per chunk, run ten warmups, and
self-check output before and after timing.

CUDA 12.6.85 compiles the runtime, test, and benchmark with warnings as errors.
The CUDA suite passes 185 host tests and registers 66 GPU tests (251 total), all
skipped without a driver/device. Therefore neither pinned-copy bandwidth nor
host/device overlap is validated. Only a measured improvement on disclosed
hardware may make the more complex pipeline a runtime default.

### M21 — Kernel fusion — Optional

Profile-driven candidates such as bias/activation, RMSNorm, or SwiGLU. Record
latency, launch count, and traffic before and after.

### M22 — Prefix cache — Implemented, CPU-validated, sanitized, and benchmarked

Implementation record (2026-08-23, commits `56d0cd0`, `cc4bc29`, `888dd30`,
`56db693`, `fbc6b20`, and `7840546`):

`PagedKvCache::fork_sequence` aliases a positive, block-aligned source prefix by
copying physical IDs into a new block table and incrementing per-page references.
Full-page alignment is the immutability contract: each branch appends only to an
exclusive tail, so divergence needs no partial-page copy-on-write. Release returns
a page to the FIFO only after its final logical owner. Debug invariants rebuild all
references, require shared pages to be full, and reconcile logical active tokens
with unique resident payload.

`PagedPrefixCache` layers exact namespace/token keys, persistent anchor sequences,
an MRU list, and bounded entry/block-reference budgets over that primitive. Its
production index uses `(namespace, aligned length, incremental 64-bit digest)` to
find candidates, then compares the entire token vector so digest collisions cannot
create false identity. Longest lookup probes every aligned prefix during one input
scan and returns an owned sequence fork. Duplicate insertions refresh recency;
oversized entries are rejected without mutation; bounded insertion, explicit
erase/clear, and LRU eviction release anchor references. The documented lock order
is prefix manager before paged cache.

`FifoScheduler::submit_prefilled` accepts a validated aligned logical seed but
does not own or discover K/V. It records progress only on FIFO admission and
schedules only the suffix. The optional CPU runtime cache looks up at most
`floor((prompt_size - 1) / block_tokens) * block_tokens`, deliberately leaving a
token to regenerate logits. After successful final prefill it publishes that
request's maximum reusable aligned prefix. If retained anchors prevent any
scheduling progress, the runtime evicts LRU entries and retries against the real
free-page count. Cancellation/failure release live forks; anchors remain until
bounded or pressure eviction.

Seven prefix-index tests cover exact longest matching, namespaces/content,
duplicate and capacity behavior, transactional invalid/stale/foreign inputs,
anchor/destructor lifetime, and concurrency. Two scheduler and two runtime tests
cover seeded admission, suffix-only execution, cancellation, output equivalence
with full recomputation, and forced pressure eviction. Debug and ASan/UBSan CPU
suites pass 169/169; the Release targets and benchmark smoke tests pass. CUDA
12.6.85/GCC 13.2 compiles every target and passes the same 169 host tests; all 57
GPU-runtime tests are explicitly skipped without an NVIDIA driver/device (226
registered tests total).

The page-sharing lifecycle benchmark remains the mechanism baseline: sharing
measured 11.02x–320.67x faster than duplicating 32 KiB–1 MiB of already-computed
payload while assigning no child payload pages. A same-binary content-index
experiment shows the honest crossover: at one entry, indexed hot hit was 136 ns
versus 81.2 ns linear; at 1,024 entries, it was 165 ns versus 1,529 ns, and indexed
miss was 33.1 ns versus 1,961 ns. The steady-state CPU runtime experiment measured
reuse 68.7% slower for 8-token prompts, 1.80x faster for 16 tokens, and 6.04x faster
for 32 tokens across the disclosed tiny fixtures. Prefix reuse remains opt-in;
these results are neither GPU nor production-model claims.

Remaining M22 limitations are explicit: CPU-only anchors, one maximum published
prefix per request, an engine-local namespace, no cached logits, no partial-page
sharing/copy-on-write, and no cross-model shared cache.

### M23 — Memory pressure and preemption — Safe admission and prefix eviction implemented; request preemption optional

The implemented CPU engine rejects submission when one request's maximum live K/V
prefix cannot fit in the physical pool. Transient pressure does not fail a request:
the scheduler preserves FIFO order, waits for terminal reclamation, then admits more
work from the cache's actual free-block snapshot. Exact-boundary and 32-request
stress tests cover progress, reuse, and leak-free exhaustion. With prefixing
enabled, a no-progress capacity result triggers measured LRU anchor eviction and a
fresh scheduling attempt. Live requests are never evicted. Request preemption
remains an optional extension requiring explicit fairness, restoration, and K/V
ownership semantics.

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
- `docs/RUNTIME.md` records M19 orchestration ownership and failure boundaries.
- `docs/INTERVIEW_GUIDE.md` evolves with implemented code and cites actual types.
- `docs/RESUME.md` contains only validated implementation and measured claims.

## Strong portfolio release gate

The first portfolio release requires excellent M0–M8 work, correct attention and
paged KV state through M13, deterministic continuous batching through M16, and a
credible implementation of M17–M19 without weakening earlier quality. If
CUDA end-to-end execution remains deferred, the release must say so plainly and
present the validated synchronous CPU engine plus compile-only CUDA layers as the
current project scope.
