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
scheduler. The current end-to-end path loads a checksummed native model format,
runs a narrowly defined Float32 Llama-style model on CPU, performs single-sequence
greedy generation, and drives multiple scheduled requests through a shared paged
KV pool with transactional prefill/decode/reclamation.
CPU paths pass normal and sanitizer suites and have reproducible benchmarks. CUDA
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

Storage can also wrap an aligned externally owned CPU range with an explicit
shared lifetime. That path is read-only: mutable `data()` access and in-place
operations fail, views preserve immutability, and `clone` returns writable owned
storage. Helix uses this contract for mapped model weights so a shallow alias
cannot silently mutate parameters.

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
cannot partially change the mapping. Release prevalidates physical references,
decrements once per table entry, and returns a page to the FIFO only at zero. A
move-only read lease pins storage/table state while direct paged attention traverses
spans.

## How does Helix share a KV prefix?

`PagedKvCache::fork_sequence` accepts a positive prefix ending exactly on a page
boundary. It copies the source's leading physical IDs into a new logical table and
increments each page's reference count; it never copies K/V payload. Since every
shared page is full, future writes touch only exclusive tail pages, which makes
divergent continuation safe without copy-on-write. Release decrements
references and returns a page to the FIFO only at zero, so either sequence may die
first. Statistics distinguish logical active tokens from unique resident tokens.

`fork_sequence` is intentionally only the ownership primitive and does not hash
prompt content by itself. `PagedPrefixCache` adds an exact namespace/token key, a
persistent anchor sequence, a most-recently-used list, and entry/block-reference
limits. Longest lookup incrementally hashes every block-aligned query prefix and
probes `(namespace,length,digest)`, but verifies the complete token vector before
accepting a candidate; collisions cannot create false reuse. A hit forks the
anchor and transfers that child handle to the caller. Bounded insertion and
pressure eviction release LRU anchors.

The CPU runtime searches only through the largest aligned prefix that leaves one
prompt token to execute, because K/V does not include the logits after the cached
prompt. It seeds scheduler progress from a hit, executes only the suffix, and
publishes the maximum reusable prefix after successful prefill. Page sharing alone
measured 11.02x–320.67x faster child lifecycle than copying 32 KiB–1 MiB. The full
runtime result is workload-dependent: reuse was 68.7% slower for the measured
8-token prompt, then 1.80x and 6.04x faster for 16 and 32 tokens. None is a GPU or
production-model claim.

## Explain continuous batching

Static batching waits for an entire cohort, so finished slots stay idle behind the
longest request. Helix removes terminal requests after each iteration and admits
waiting requests into those slots before surviving requests finish. Scheduler
tests demonstrate `A B C D`, then after A finishes, `B C D E`. Token and logical
KV-block budgets still constrain each batch. A synthetic constant-cost policy
model shows the utilization effect but is explicitly not measured inference
throughput. Separately, the M19 CPU engine test executes the replacement pattern
with real model and paged-cache work; its benchmark measures serial CPU trace
throughput without claiming GPU-style batched execution.

## How is prefill different from decode?

Prefill processes prompt tokens, usually with many query positions and quadratic
causal attention within the prompt. Decode produces one new position per request
and reads a growing cached K/V history, so matrix shapes are much narrower and
often memory/launch-latency sensitive. Helix's CPU generator runs full prompt
prefill once, stores each layer's rotated keys and values, then computes only the
new token position on decode. Deterministic tests compare successive cached logits
and greedy tokens with full-prefix recomputation. The M19 CPU engine additionally
executes mixed scheduler work against per-request paged histories. Complete new
prompts use full prefill plus bulk page append; chunked prompts use the retained
one-token path. Work descriptions still execute serially, so this is continuous
orchestration rather than a fused tensor batch.

## What limits inference throughput?

It depends on phase, batch, precision, and hardware: projection/MLP GEMMs,
attention bandwidth and workspace, KV traffic/capacity, launch overhead, allocator
or scheduling bubbles, and CPU orchestration can all dominate. Helix's CPU
prefill/decode fixtures show the complete reference-path cost but do not isolate a
production bottleneck; CUDA profiling is required before naming the GPU limiter.

## What limits decode latency?

Decode has small per-request GEMMs, one query against a growing KV history,
launch/synchronization overhead, KV memory bandwidth, and batching delay. Increasing
batch can improve throughput while hurting an individual request's queueing or
inter-token latency. Helix's scheduler metrics separate wait, TTFT, request
latency, and adjacent-token intervals so that tradeoff can eventually be measured.
On the current CPU baseline, removing an intermediate K/V snapshot cut isolated
decode staging by 45.4%–58.1% for the measured multi-token histories, but complete
decode remains dominated by model math as dimensions grow.

## What happens when KV memory is exhausted?

The current policy is safe non-admission. `PagedKvCache::append` fails
transactionally when a new block is required and the pool is empty; an existing
partially filled block remains writable. The scheduler accepts a free-block
snapshot, keeps a FIFO head waiting when it cannot reserve the required delta, and
does not pretend it owns physical blocks. A final length-limited output needs no
new K/V because it is sampled from existing logits, so it can retire and release
an exactly full block even with zero free pages. The CPU engine tests that boundary
and real block reuse. If optional prefix anchors are the reason no batch can be
formed, the CPU engine evicts LRU anchors and retries with the cache's new free-
block count. It never evicts a live request. Request preemption is not implemented.

## What would break first at 10x workload?

On the current CPU model, single-threaded reference GEMM would dominate quickly;
prefill also carries an `O(T^2)` score workspace. Paged one-token decode now avoids
history compaction, but still traverses all prior K/V, allocates a linear score
workspace, and holds a coarse cache-wide read lease per layer. The engine executes
scheduled work serially rather than forming tensor batches, and the scheduler
retains terminal request history. For KV, fixed pool
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

## How does Helix convert and validate model weights?

Runtime loading uses the C++ `HELIXLM1` format only: a fixed little-endian header,
canonical tensor directory, 64-byte payload alignment, exact file coverage,
finite Float32 values, and FNV-1a metadata/payload checksums. FNV is accidental-
corruption detection, not authentication. Both native loaders share the same
metadata parser and return no partial model after an error. The portable path
decodes into aligned owned Tensors. The opt-in POSIX path maps a regular file
read-only and exposes payload-sized immutable Tensor views that share one RAII
mapping lifetime; it still scans every byte for checksum and finite-value
validation.

Memory mapping is not a snapshot or a universal portability layer. It requires a
little-endian IEEE-754 POSIX host, and the source inode must not be modified or
truncated while the model lives. Rename/unlink is safe after mapping. The copied
loader remains the fallback for streams and other hosts. In a paired warm-page
fixture, mapping reduced median CPU load time by 12.78%–20.96% across three
disclosed synthetic files; that is not cold-disk latency or an RSS result.

The optional offline converter accepts one narrow Hugging Face Llama safetensors
layout. Python/NumPy strictly parses single or indexed shards, validates the exact
architecture, converts F32/F16/BF16 to Float32, and transposes PyTorch
`[output,input]` projections into Helix `[input,output]`. It also interleaves Q/K
coordinates within each head because Hugging Face Llama uses half-split RoPE while
Helix rotates adjacent pairs. A C++ packer—not Python—owns native file checksums
and canonical serialization. Packing uses private staging plus atomic replacement,
so `--force` does not destroy an old output before the replacement is complete.

The strongest test is semantic, not just structural: a synthetic sharded mixed-
precision checkpoint is evaluated by an independent source-layout NumPy oracle,
then its final logits are compared with C++ Helix execution after conversion.
That does not justify claiming every public Llama-family checkpoint; GQA/MQA,
biases, scaled/partial RoPE, quantization, adapters, tokenization, and downloaded-
checkpoint validation remain outside the supported boundary.

## What did Helix implement itself versus delegate to libraries?

Helix implements Tensor/storage metadata, CPU kernels, CUDA elementwise/reduction/
softmax/GEMM/transformer/attention kernels, allocator metadata and deferred reuse,
KV layouts, paged attention, requests, scheduling/metrics, the CPU transformer,
cached generation session, exact bounded prefix index/LRU, paged multi-request CPU
engine with suffix reuse, greedy selection, native file parser/packer, and the
narrow safetensors metadata/layout converter. It also implements read-only
external Tensor ownership and the checked POSIX model-mapping path. It delegates
numerical array transposition in the one-time converter to NumPy, CUDA driver/
runtime operations to the CUDA Runtime API, and testing/benchmark harnesses to
GoogleTest/Google Benchmark.
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

The same discipline helped decode: the first path copied cache history into an
owning snapshot and then copied it again into the attention buffer. A paired
fixture kept that baseline and showed direct prefix copy reducing staging medians
by 45.4%–58.1% on the multi-token cases while preserving locking and stale-length
checks. I do not quote those percentages as an end-to-end model speedup because
the model benchmark contains much more work.

M19 used the same paired approach at the orchestration boundary. The retained
baseline evaluated every prompt token through one-token paged decode; the default
executes a complete new prompt once and bulk-appends its layer-major K/V. Across
the three disclosed tiny CPU traces this reduced median end-to-end trace time by
15.8%–51.3%. The gain narrowed as decode dominated, which is more informative than
presenting one favorable small case. Construction, submission, and final reporting
were excluded; scheduler, model, cache, completion, and reclamation were timed.

A follow-up result was more nuanced. Direct paged decode removed two combined K/V
tensors and cut the isolated attention-layer median by 13.3%–27.7%, but complete
runtime traces improved by only 0.08%–6.23%; the largest case was effectively a
tie relative to variation. That taught me to distinguish a verified local
bottleneck from overall model throughput. Helix keeps `kCompactCopy` selectable so
the comparison remains reproducible.

Mapped model loading was a bounded win rather than “zero-copy means instant.” It
removed owned payload allocation and copying, but retained full checksum and
finite-value scans. Warm-page median CPU time improved 20.96%, 15.14%, and 12.78%
as file size grew. The narrowing result makes the remaining linear validation cost
visible and is why I do not describe mapping as constant-time loading, cold-disk
speed, or a measured memory reduction.

Prefix reuse produced the clearest negative result. It cut executed prompt tokens
in half for the eight-token fixture but made the complete trace 68.7% slower:
hashing, page-reference bookkeeping, and four sequential suffix calls cost more
than one vectorized full prefill. The same implementation became 1.80x and 6.04x
faster at 16 and 32 tokens. That is why the cache is opt-in and why I would profile
a representative prompt distribution before adding an activation threshold.
