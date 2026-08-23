# Helix Interview Guide

This guide describes the repository as implemented. “CUDA compile-validated” means
NVCC built the path; it does not mean a kernel ran on NVIDIA hardware. Recheck the
roadmap and performance notebook before quoting a test count or measurement.

## Give me a 60-second explanation of Helix

Helix is a C++20/CUDA inference runtime I built to expose the systems mechanisms
normally hidden by a framework. It has a checked Tensor and storage layer,
single-threaded CPU correctness kernels, manually written CUDA elementwise,
reduction, softmax, GEMM, transformer, and attention kernels, a caching allocator
with fragmentation metrics and deferred stream-ordered reuse, contiguous and paged
KV caches, direct block-table attention, and a deterministic continuous-batching
scheduler. The current end-to-end math path runs a narrowly defined Float32
Llama-style full prefill on CPU and loads a checksummed native model format. CPU
paths pass normal and sanitizer suites and have reproducible benchmarks. CUDA
paths compile under CUDA 12.6, but I do not claim GPU correctness or performance
because this development environment has no NVIDIA device.

## Why build an inference runtime rather than use PyTorch?

PyTorch is the right answer for many applications, but it intentionally hides
details this project is meant to study: tensor/storage ownership, overflow-safe
shape arithmetic, launch and stream semantics, allocator reuse, physical KV
layout, scheduler resource accounting, and the performance effect of loop or
memory order. Helix keeps a small explicit implementation so each mechanism can be
tested and measured. It is not an argument that a learning runtime should replace
PyTorch in production.

## How is a Tensor represented internally?

`helix::Tensor` holds signed 64-bit shape and strides, dtype, device, element and
byte counts, a storage byte offset, and shared ownership of `Storage`. Layout is
currently canonical contiguous row-major. Copies are shallow handles; `clone` and
device transfer allocate independent storage; reshape/view retain shared storage
after checked bounds and element-count validation. Shape and byte multiplications
reject negative values and overflow before pointer arithmetic or allocation.

## Why is GPU memory allocation expensive?

Direct device allocation can enter the CUDA runtime/driver, update global device
memory bookkeeping, and impose synchronization costs depending on API and context.
Doing that for every temporary also prevents cheap reuse and makes latency
variable. The exact cost is hardware/runtime dependent; Helix has CUDA comparison
fixtures but has not executed them, so it does not quote a GPU allocation speedup.

## Explain HelixAllocator

`HelixAllocator` obtains larger segments from a pluggable `MemoryResource` and
suballocates aligned blocks. It scans free blocks for the candidate with the least
remainder after alignment, splits prefix/allocated/suffix pieces, and merges
adjacent same-segment free blocks on release. Handles carry allocator identity,
allocation ID, and generation to reject foreign handles, double frees, and stale
ABA-style reuse. A mutex protects the metadata boundary. Statistics include
reserved, requested, rounded-active, free, peak, largest-free-block, reuse, and
upstream-call counts.

Best fit was selected as an auditable baseline for mixed lifetimes, not assumed to
be universally optimal. Its linear scan and metadata cost remain visible in host
benchmarks.

## What causes memory fragmentation?

Different allocation sizes and lifetimes leave holes. A request can fail even when
aggregate free bytes are sufficient if no single aligned free block is large
enough. Alignment and size rounding also consume bytes that the client did not
request. Long-lived weights/KV state interspersed with short-lived temporaries are
a typical inference pattern that can make both effects visible.

## Internal versus external fragmentation?

Helix defines internal fragmentation as rounded active bytes minus client-requested
active bytes. It defines external fragmentation as
`1 - largest_free_block / total_free_bytes`, or zero when there are no free bytes.
The first measures waste inside active allocations; the second measures how split
the free address space is. Pool reservation beyond current demand is reported
separately rather than hidden in either number.

## Why can't freed GPU memory always immediately be reused?

Kernel launches are asynchronous. Returning from a launch does not mean the stream
has stopped reading or writing its buffers. Reassigning an address before all
users are ordered complete creates use-after-free-style corruption on the device.
`DeferredAllocator` therefore retains a logically freed block behind a completion
object. `StreamOrderedAllocator` records a CUDA event on the explicit release
stream and only makes the block reusable after the event completes. This is safe
only if that stream is ordered after every use; Tensor does not yet track arbitrary
cross-stream use, so general allocator integration remains pending.

## What are CUDA streams?

A stream is an ordered queue of device work. Operations in one stream execute in
issue order; different streams may overlap when dependencies and hardware permit.
Host launch return is normally asynchronous. Events create observable completion
points and cross-stream dependencies. Helix wraps streams/events in move-only RAII,
checks device matching and CUDA errors, and times GPU work with same-stream events
rather than host launch duration.

## Explain global memory versus shared memory versus registers

Global memory is large device DRAM visible across blocks and relatively high
latency. Shared memory is a programmer-managed on-chip allocation shared by threads
in one block, useful for reusing tiles but requiring barriers and careful bank
access. Registers are per-thread, lowest-latency state, but high register demand
can reduce resident warps or spill into local memory backed by global memory. The
right placement follows reuse and access patterns, not a blanket “faster memory is
always better” rule.

## What is memory coalescing?

Threads in a warp should access addresses that the memory system can serve with a
small number of aligned transactions. Helix elementwise kernels use grid-stride
loops over consecutive elements. Tiled GEMM loads adjacent matrix elements into
shared memory. RoPE assigns threads to adjacent coordinate pairs. Paged attention
needs additional care because logical neighbors can cross physical pages; the CPU
path proves semantics, while GPU coalescing analysis remains pending a device
implementation and profiler run.

## Explain warp execution

NVIDIA schedules threads in groups of 32 called warps. A warp issues an instruction
for its active lanes; divergent branches serialize paths for different lane masks.
Warp shuffle instructions exchange register values without shared-memory round
trips. Helix's warp-assisted reduction uses shuffles within each warp and a small
shared-memory handoff across warp results. That code is compile-validated but not
runtime-profiled here.

## Walk me through your GEMM kernel

The naive kernel gives each thread one logical output element and loops over K,
with grid-stride traversal when the matrix exceeds the capped two-dimensional
grid. The tiled kernel uses a 16x16 block. Threads cooperatively load one A tile
and one B tile into shared memory, zero-fill edge lanes, synchronize, accumulate
the tile product in a register, synchronize before overwriting the shared tiles,
and finally store in-bounds outputs. Both support odd dimensions and zero inner
extent and are tested against a double-accumulating CPU oracle. No cuBLAS path is
currently implemented.

## Why does tiling improve GEMM?

Without tiling, many threads repeatedly fetch the same A and B values from global
memory. A shared tile allows each loaded value to participate in multiple
multiply-adds, increasing arithmetic intensity. It also regularizes coalesced
loads. The benefit is not claimed until the kernels run on real hardware; current
evidence is compilation, edge correctness design, and static resource inspection.

## What is occupancy?

Occupancy is active warps divided by the architecture's maximum active warps per
SM. Registers, shared memory, threads per block, and architectural limits bound it.
It is a latency-hiding capacity metric, not direct useful-work throughput.

## Why can higher occupancy sometimes not improve performance?

Once there are enough warps to hide the relevant latency, more occupancy may not
help. Reducing register use to force occupancy can cause spills; smaller tiles can
lower data reuse; and bandwidth- or instruction-throughput-bound kernels may not
benefit from more resident warps. Helix records compiler resource data as static
evidence only and reserves an occupancy conclusion for profiling on hardware.

## What is KV cache?

Autoregressive attention needs prior tokens' keys and values at every decode step.
Those projections do not change, so caching them avoids recomputing K/V for the
whole prefix. Helix tests this by appending prior K/V and showing one-token decode
attention matches attention over the equivalent complete history.

## Why does LLM inference need so much KV memory?

For Float32 contiguous Helix storage, one sequence reserves
`layers * 2 * kv_heads * capacity_tokens * head_dimension * 4` bytes. The factor
two is keys plus values. Production values are large because capacity grows with
sequence length and the product includes every layer and concurrent request. Lower
precision reduces element bytes but not the linear scaling factors.

## Explain paged KV cache

Instead of allocating every request's maximum contiguous history, `PagedKvCache`
owns fixed-token physical K/V blocks. A sequence acquires blocks only when append
crosses a logical block boundary. This bounds per-sequence tail waste and lets
finished sequences return blocks to one global FIFO pool. It trades contiguous
addressing for a block-table lookup. CPU measurements preserve both sides: paging
reduced selected lifecycle costs, while compacting large histories was slower.

## How does your block table work?

Each cache-tagged sequence owns an ordered vector of `PhysicalBlockId` values.
Logical position `p` maps to table entry `p / block_tokens` and in-block offset
`p % block_tokens`. The physical pool is laid out
`[block, layer, head, token_in_block, dimension]`. Appending reserves table
capacity before consuming a free block so metadata allocation or block exhaustion
cannot partially change the mapping. Release prevalidates ownership, then returns
every block exactly once. A move-only read lease pins storage/table state while
direct paged attention traverses spans.

## Explain continuous batching

Static batching waits for an entire cohort, so finished slots stay idle behind the
longest request. Helix removes terminal requests after each iteration and admits
waiting requests into those slots before surviving requests finish. Scheduler
tests demonstrate `A B C D`, then after A finishes, `B C D E`. Token and logical
KV-block budgets still constrain each batch. A synthetic constant-cost policy
model shows the utilization effect but is explicitly not measured inference
throughput.

## How is prefill different from decode?

Prefill processes prompt tokens, usually with many query positions and quadratic
causal attention within the prompt. Decode produces one new position per request
and reads a growing cached K/V history, so matrix shapes are much narrower and
often memory/launch-latency sensitive. Helix currently has full CPU model prefill,
cache/attention decode primitives, and scheduler work kinds, but has not connected
them into M18 generation yet.

## What limits inference throughput?

It depends on phase, batch, precision, and hardware: projection/MLP GEMMs,
attention bandwidth and workspace, KV traffic/capacity, launch overhead, allocator
or scheduling bubbles, and CPU orchestration can all dominate. Helix's current CPU
full-prefill benchmark composes the path but does not isolate a bottleneck; CUDA
profiling is required before naming the GPU limiter.

## What limits decode latency?

Decode has small per-request GEMMs, one query against a growing KV history,
launch/synchronization overhead, KV memory bandwidth, and batching delay. Increasing
batch can improve throughput while hurting an individual request's queueing or
inter-token latency. Helix's scheduler metrics separate wait, TTFT, request
latency, and adjacent-token intervals so that tradeoff can eventually be measured.

## What happens when KV memory is exhausted?

The current policy is safe non-admission. `PagedKvCache::append` fails
transactionally when a new block is required and the pool is empty; an existing
partially filled block remains writable. The scheduler accepts a free-block
snapshot, keeps a FIFO head waiting when it cannot reserve the required delta, and
does not pretend it owns physical blocks. Preemption and prefix eviction are not
implemented.

## What would break first at 10x workload?

On the current CPU model, single-threaded reference GEMM and full-history prefill
would dominate quickly, followed by temporary allocations and attention's
`O(T^2)` score workspace. In serving metadata, the scheduler retains terminal
request history and several operations are linear scans. For KV, fixed pool
capacity becomes the admission boundary. On GPU, the answer requires profiling;
likely constraints include KV capacity/bandwidth, GEMM efficiency, allocator/Tensor
integration, and missing batched executor plumbing.

## How would this architecture change across multiple GPUs?

Start with explicit device ownership and shard a linear layer by output or input
features. Column-parallel projections require a later gather or compatible next
shard; row-parallel projections require an all-reduce of partial outputs. KV and
attention ownership should follow head partitioning where possible. The scheduler
must admit only when every participating rank has capacity, and failures must be
coordinated. Helix intentionally defers this until one-GPU execution is validated.

## Why NCCL?

NCCL provides topology-aware GPU collectives such as all-reduce and all-gather and
integrates them with CUDA streams. It is preferable to hand-building production
collectives for the optional tensor-parallel demonstration. Helix does not yet
enable NCCL or claim multi-GPU support.

## What did Helix implement itself versus delegate to libraries?

Helix implements Tensor/storage metadata, CPU kernels, CUDA elementwise/reduction/
softmax/GEMM/transformer/attention kernels, allocator metadata and deferred reuse,
KV layouts, paged attention, requests, scheduling/metrics, the CPU transformer,
and the native file parser. It delegates CUDA driver/runtime operations to the
CUDA Runtime API and testing/benchmark harnesses to GoogleTest/Google Benchmark.
No PyTorch runtime, BLAS, cuBLAS, NCCL, or external model graph executes Helix
inference today. cuBLAS and NCCL remain explicitly labeled future library paths.

## What performance optimization surprised you?

Direct paged attention initially appeared faster than contiguous attention, but
the comparison was confounded: the paged value loop used a better traversal. A
32-coordinate stack tile reduced repeated page walks and improved paged CPU time
2.36x–2.99x. Applying the same tile to contiguous attention improved it too; the
fair optimized comparison showed paging 2.1%–6.7% slower across the measured
cases. The useful result was not “paging is faster,” but that paired baselines are
necessary to separate a general locality improvement from the cost of paging.
