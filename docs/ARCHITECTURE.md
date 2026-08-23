# Helix Architecture

## Purpose and current status

Helix is a lightweight C++20/CUDA runtime for studying and implementing the core
systems behind decoder-only LLM inference. It deliberately exposes tensor layout,
kernel dispatch, GPU memory lifetime, KV paging, scheduling, and asynchronous
execution rather than delegating the whole runtime to a framework.

This document is the target architecture. A component is not complete merely
because it appears here. `docs/ROADMAP.md` and the README distinguish implemented,
validated, and planned work.

## Design principles

- CPU execution is a first-class correctness path and development mode.
- CUDA is an optional compile-time backend, never a configuration-time requirement
  for CPU development.
- Ownership and asynchronous lifetime are explicit.
- Reference implementations precede optimized kernels.
- Custom kernels remain available beside any later library-backed implementation.
- Scheduler policy is independent of tensor and CUDA mechanics.
- Metrics are measurements, not feature claims.

## System context

```mermaid
flowchart TD
    Request[Inference Request] --> Scheduler
    Scheduler --> Batch[Iteration Batch]
    Batch --> Executor[Model Executor]
    Executor --> Tensor[Tensor Runtime]
    Executor --> Allocator[HelixAllocator]
    Executor --> Kernels[CPU / CUDA Kernels]
    Executor --> KV[KV Cache]
    KV --> GPUMemory[(GPU Memory)]
    Allocator --> GPUMemory
    Kernels --> GPUMemory
    Executor --> Token[Generated Token]
    Token --> Scheduler
```

The eventual runtime executes one scheduling iteration at a time: admit requests,
construct prefill/decode work within token and KV-memory budgets, execute the model,
append K/V state, select tokens, and retire or requeue requests.

## Layering and dependency direction

```mermaid
flowchart TB
    App[CLI / Examples / Benchmarks]
    Runtime[Runtime Engine]
    Model[Model Executor]
    Schedule[Request + Scheduler]
    Attention[Attention + KV Cache]
    Ops[Tensor Operations]
    Tensor[Tensor + Storage]
    Backend[CPU Backend | Optional CUDA Backend]
    Core[Core: errors, logging, dtype, device, checked sizes]

    App --> Runtime
    Runtime --> Model
    Runtime --> Schedule
    Model --> Attention
    Model --> Ops
    Attention --> Ops
    Attention --> Tensor
    Ops --> Tensor
    Tensor --> Backend
    Backend --> Core
    Tensor --> Core
    Schedule --> Core
```

Dependencies point downward. In particular, the scheduler must not include CUDA
headers, and kernels must not know about requests. Backend-specific implementation
details stay out of public, backend-neutral headers where possible.

## Build modes

### CPU-only

`HELIX_ENABLE_CUDA=OFF` configures a C++-only project. It builds core utilities,
tensor/storage, CPU kernels, request/KV metadata, scheduler logic, model structure,
and all applicable tests and benchmarks. CUDA headers and libraries are not
required.

### CUDA

`HELIX_ENABLE_CUDA=ON` explicitly requires a CUDA compiler and toolkit. It adds
CUDA sources, runtime wrappers, GPU storage, kernels, and hardware-gated tests. A
missing toolkit is a clear configuration error rather than a silent fallback.

CUDA validation has three distinct levels:

1. CPU CI: CUDA is disabled; no CUDA claim is made.
2. CUDA compile validation: CUDA translation units compile and link without a GPU.
3. GPU runtime validation: correctness, sanitizer, and benchmark workloads run on
   named NVIDIA hardware.

## Core types and error policy

`Device` is a value type containing a backend kind (`CPU` or `CUDA`) and a device
index. CPU currently permits only index zero. CUDA device construction does not by
itself touch the runtime; backend operations validate availability.

`DType` identifies the element representation and maps to a checked byte size.
Float32 is the first executable type. Float16 is introduced only with explicit
conversion and arithmetic behavior.

Recoverable host-side contract and runtime failures throw `helix::Error` with a
stable category and contextual message. Programmer invariants that cannot safely
continue use a fatal check. Destructors never throw; cleanup failures are logged or
handled by an explicit `close`/synchronization operation when callers need them.

Size helpers reject negative dimensions and detect overflow before allocation or
pointer arithmetic.

## Tensor runtime

### Representation

A `Tensor` is a small handle with:

- shape and strides,
- dtype and device,
- logical element count,
- visible byte count,
- byte offset into storage,
- shared ownership of a `Storage` allocation.

```mermaid
classDiagram
    class Tensor {
      +Shape shape
      +Strides strides
      +DType dtype
      +Device device
      +numel() size_t
      +nbytes() size_t
      +data() void*
      +reshape(Shape) Tensor
    }
    class Storage {
      +Device device
      +size_bytes size_t
      +data() void*
    }
    Tensor "many views" --> "1 shared allocation" Storage
```

Storage owns memory; Tensor owns metadata. A view shares storage and changes only
metadata/offset after validating bounds. The initial implementation supports
contiguous row-major tensors. General strided execution is deferred until kernels
need it; APIs reject unsupported non-contiguous inputs rather than silently
producing wrong results.

### Ownership

- Owning factory operations allocate `Storage` through a backend-specific storage
  routine. CUDA storage remains direct until the cache is stream-safe.
- Views retain shared storage ownership.
- A non-owning external-buffer tensor, if added, requires an explicit lifetime
  token/deleter; it will not masquerade as owning memory.
- Copies are shallow handle copies. `clone` performs a deep copy.
- Device transfer returns new storage on the target device.

## Device abstraction and backends

Backend-neutral code dispatches using the tensor's `DeviceType`. CPU storage uses
aligned host allocation. CUDA storage eventually uses the active allocator rather
than direct `cudaMalloc` per tensor.

```mermaid
flowchart LR
    TensorOp[Tensor operation] --> Dispatch{device}
    Dispatch -->|CPU| CPU[CPU reference kernel]
    Dispatch -->|CUDA enabled| CUDA[CUDA kernel / library path]
    Dispatch -->|CUDA disabled| Error[Unsupported-device error]
```

Dispatch is initially explicit and readable. A registry is justified only if the
operation/type/backend matrix becomes difficult to maintain.

## CPU backend

CPU kernels are deterministic correctness oracles and a usable fallback, not an
attempt to replace optimized BLAS. Initial Float32 kernels use straightforward
loops and numerically stable formulas. Parallelism and SIMD may be explored only
after baselines are retained and measured.

The reference set grows in dependency order: elementwise operations, activations,
reduction, softmax, matrix multiplication, RMSNorm, SwiGLU, adjacent-pair RoPE,
and bias-free linear projection.

## CUDA backend

### Runtime layer

The CUDA layer wraps device selection, streams, events, allocations, copies, and
synchronization in move-only RAII types. Every CUDA API call produces contextual
errors. Public headers either avoid CUDA types or hide them behind backend-specific
headers so CPU clients do not require the toolkit.

### Kernel progression

Educational kernels remain separately selectable and benchmarkable:

- elementwise: direct one-element-per-thread baselines,
- reductions: naive, shared-memory, then warp-assisted variants,
- softmax: correct row-wise baseline, then reduced global traffic,
- GEMM: naive, shared-memory tiled, then evidence-driven refinements,
- transformer operations: RMSNorm, RoPE, SwiGLU, and attention components.

cuBLAS/cuBLASLt may later provide an optimized GEMM path. It will be labeled as a
library backend and compared honestly with custom kernels.

The implemented transformer primitives retain the same separation. CUDA RMSNorm
uses a block/warp reduction, SwiGLU uses a pointwise grid-stride kernel, and RoPE
maps one thread to each adjacent coordinate pair. RoPE input is
`[tokens, ..., head_dimension]`, with preselected cosine/sine rows shaped
`[tokens, head_dimension/2]`; this keeps position selection outside the math kernel
and supports packed nonuniform positions later. Linear projection is a named
adapter over the selectable custom SGEMM and uses row-major
`[input_features, output_features]` weights.

## Memory allocator

`HelixAllocator` owns segments obtained from a pluggable `MemoryResource` and
suballocates aligned blocks. `SystemMemoryResource` makes the complete metadata
algorithm executable under CPU-only builds; `CudaMemoryResource` is the direct
`cudaMalloc`/`cudaFree` adapter. The baseline uses best-fit selection because LLM
workloads mix long-lived allocations with transient buffers, but benchmarks must
test that hypothesis rather than assuming it.

Each segment stores a vector of address-ordered blocks. The current baseline scans
all free blocks linearly, including alignment padding, then selects the candidate
with the smallest remainder and deterministic segment/address tie breaking. This
is intentionally simpler and easier to audit than a second size index; profiling
must justify a more complex index. Requested sizes round to the effective power-of-
two alignment, and allocation splits exact prefix/suffix remainders. Freeing
rebuilds the affected address-ordered list and merges every adjacent free block in
the same segment.

```mermaid
flowchart LR
    Need[Aligned request] --> Find[Best-fit free block]
    Find --> Split{Useful remainder?}
    Split --> Prefix[Optional alignment prefix]
    Split --> Used[Generation-tagged allocated block]
    Split --> Suffix[Optional free suffix]
    Release[Validated allocation handle] --> Merge[Merge adjacent free blocks]
    Merge --> Cache[Reusable cached segment]
    Cache --> Trim[Explicitly release wholly free segment]
```

The allocator reports reserved, rounded allocated, requested, free, peak, largest
free block, active/segment counts, cache reuse, and upstream allocation/release
counts. External fragmentation is `1 - largest_free_block / bytes_free` when free
bytes are nonzero. Internal fragmentation is
`(bytes_allocated - bytes_requested) / bytes_allocated` when allocated bytes are
nonzero. Both are otherwise zero.

Allocation handles carry a process-unique allocator ID and monotonic local
allocation ID as well as pointer/size/alignment/device. Deallocation validates all
fields and both identities, so neither a stale copy nor a colliding handle from
another pool can free an allocation (the raw-pointer ABA/cross-pool problem). A mutex
serializes metadata and upstream calls. Debug builds re-derive segment coverage,
block adjacency, counters, active mappings, and alignment after every mutation.

`DeferredAllocator` prevents immediate free-list reuse by holding an allocation in
a pending state behind a `Completion`; pending blocks remain active in the base
pool. `StreamOrderedAllocator` supplies CUDA-event completions recorded on an
explicit same-device release stream. Reclamation is nonblocking when polling and
blocking during explicit drain/destruction. This adapter is compile-validated, not
GPU-runtime-validated.

Tensor storage is not wired into the cache yet. A release event is safe only when
all prior uses happen-before that event. Tensor currently has no per-storage stream
ledger and therefore cannot construct the required dependency chain for
multi-stream use automatically. CPU completion tests are not presented as proof of
CUDA event behavior.

## Attention

The implemented contiguous scaled dot-product attention path materializes the
understandable stages:

```text
scores = Q * transpose(K)
scores = masked(scores / sqrt(head_dimension))
probabilities = softmax(scores)
output = probabilities * V
```

Q/K/V layouts are `[heads, query_tokens, head_dimension]`,
`[heads, kv_tokens, head_dimension]`, and
`[heads, kv_tokens, value_dimension]`. An explicit query-position offset defines
causal masking for prefill and decode. The caller-owned score workspace is
`[heads, query_tokens, kv_tokens]`, or exactly `4*H*Q*K` bytes for Float32; softmax
reuses it in place.

The CPU implementation is the double-accumulating oracle. Its original
probability-times-V loop remains as a dimension-major benchmark baseline. The
default path uses a fixed 32-coordinate stack tile so each contiguous V row is
consumed while hot without changing any coordinate's accumulation order. The CUDA
baseline uses one serial-dot-product thread per score, the existing stable block
softmax, and one serial-KV-reduction thread per output coordinate. This deliberately
visible path establishes correctness and memory-cost baselines. Q/K/V heads must
currently match; GQA is not implied.

The CPU paged path accepts a move-only layer read lease from `PagedKvCache` and
resolves logical blocks directly to per-head K/V spans. It never materializes a
compact K/V history:

```mermaid
flowchart LR
    Query[Query rows] --> Scores[Scaled causal scores]
    Lease[Layer read lease] --> Table[Logical block table]
    Table --> KeyPages[Physical key spans]
    Table --> ValuePages[Physical value spans]
    KeyPages --> Scores
    Scores --> Softmax[Stable in-place softmax]
    Softmax --> Values[32-coordinate value tiles]
    ValuePages --> Values
    Values --> Output
```

The lease shares ownership of the cache implementation and holds its mutex from
construction through destruction. That pins both the physical pool and the
sequence table and excludes append/release while raw spans are in use. It is safe
but intentionally coarse: one long read blocks unrelated sequences in the same
cache. The final logical block exposes only its written prefix, so stale bytes in
unused slots or recycled blocks are unreachable. CPU measurements compare retained
dimension-major baselines, equal tiled traversals, and the page-table overhead.
There is no CUDA paged-attention implementation yet; CPU locality results are not
GPU coalescing evidence.

## KV cache

For each layer, cached keys and values avoid recomputing projections for prior
tokens during decode. The implemented Float32 CPU baseline stores two owned tensors
per sequence with layout `[layers, kv_heads, capacity_tokens, head_dimension]`.
Appending one `[layers, kv_heads, head_dimension]` token fills the current logical
position. Layer retrieval creates a compact owning
`[kv_heads, active_tokens, head_dimension]` snapshot compatible with contiguous
attention; a preallocated-output form excludes allocation when required.

`ContiguousKvCache` is the sole storage owner. Non-owning sequence handles combine
a process-unique cache ID with a monotonic local sequence ID, so foreign, stale,
and released handles cannot mutate another allocation. Creation reserves the full
capacity under a byte ceiling, and exhaustion is transactional. One mutex
serializes mutable sequence state; returned snapshots are independent. Exact
layout, byte formulas, tests, and limitations are documented in
`docs/KV_CACHE.md`.

Paged KV storage divides capacity into fixed-token blocks. A per-sequence block
table maps logical token blocks to physical blocks:

```mermaid
flowchart LR
    Pos[Logical token position] --> Logical[block = pos / block_tokens]
    Logical --> Table[Sequence block table]
    Table --> Physical[Physical BlockId]
    Physical --> Offset[offset = pos % block_tokens]
    Offset --> KV[(K/V block storage)]
```

The implemented M12 CPU manager owns two physical-major tensors shaped
`[physical_blocks, layers, kv_heads, block_tokens, head_dimension]`, a preallocated
circular FIFO free-block pool, owner/free/reuse metadata, and one block table per
sequence. It allocates a physical block only when an append begins a new logical
block. The sequence table is grown before the free queue changes, so metadata OOM
or pool exhaustion preserves the prior mapping and logical length. Release
prevalidates ownership and returns blocks without allocating.

The first memory-pressure policy rejects append with a typed resource error while
preserving current sequences; an append within an already-owned partial block can
still proceed. Debug builds re-derive ownership and accounting invariants after
every mutation. Both contiguous and paged CPU caches use one mutex per instance.
The M13 read lease lets attention traverse written block spans without compaction;
its exclusive lifetime lock is the current invalidation boundary. GPU-resident pool
integration, preemption, prefix sharing, and reference counting remain separate
milestones rather than implied features.

## Request and scheduler

The implemented M14 request is a move-only owner of a strong nonzero request ID,
immutable prompt tokens, generated-token/timestamp records, generation limit,
terminal context, and validated state machine:

```mermaid
stateDiagram-v2
    [*] --> Waiting
    Waiting --> Prefill
    Prefill --> Decoding
    Prefill --> Finished
    Decoding --> Decoding
    Decoding --> Finished
    Waiting --> Cancelled
    Prefill --> Cancelled
    Decoding --> Cancelled
    Waiting --> Failed
    Prefill --> Failed
    Decoding --> Failed
    Finished --> [*]
    Cancelled --> [*]
    Failed --> [*]
```

Positive prefill progress may be chunked across iterations. Completing the prompt
enters decode, except a zero generation budget finishes immediately. Appending a
generated token at the configured limit finishes with `length`; EOS, cancellation,
and failure are explicit alternatives. Terminal states have no exits. Every event
uses a caller-supplied monotonic `steady_clock` time, allowing deterministic wait,
TTFT, completion, and inter-token metric derivation. The request is not thread-safe;
one scheduler thread owns mutation. Details and the exhaustive transition matrix
are in `docs/SCHEDULER.md`.

The implemented FIFO scheduler owns requests in a stable map, orders waiting IDs in
a deque, and bounds running IDs in admission order. Each iteration is limited by a
token budget, running-request capacity, and a caller-supplied count of free KV
blocks. Prefill describes a positive prompt range and may be chunked; decode emits
one-token work. Existing running requests are considered before waiting admission,
which is deterministic but intentionally permits head-of-line delay.

```mermaid
sequenceDiagram
    participant KV as KV authority
    participant S as FIFO scheduler
    participant E as Executor
    KV->>S: free-block snapshot
    S-->>E: batch(id, token ranges, new blocks)
    E->>KV: reserve reported blocks
    E-->>S: ordered work completions
    S-->>KV: terminal block releases
```

The scheduler does not own physical cache blocks. It predicts block deltas from
logical token counts; the runtime must apply reservations before execution and
apply release counts afterward. Only one batch can be in flight. Completion first
validates every ordered result, then stages all potentially allocating state before
request mutation. This prevents a malformed late result or allocation failure from
partially committing an iteration.

Scheduler state remains host-only and deterministic in tests. It is not thread-
safe; the runtime engine will own it from one orchestration thread. A future multi-
producer submission queue may provide the synchronization boundary.

Continuous replacement is a consequence of the completion/admission boundary:
terminal IDs leave the running vector after an iteration, and the next call admits
waiting IDs into those slots before the surviving cohort finishes. Mixed batches
may therefore carry decode work for resident requests and prefill ranges for newly
admitted requests. This is implemented and deterministically tested at the host
policy layer; the model executor integration is still pending.

Metrics are derived from immutable request snapshots at a caller-provided
`steady_clock` observation time. The reporting path validates distinct live IDs and
monotonic time, aggregates state/token counters, then sorts duration samples for
linearly interpolated p50/p95 values. Waiting time includes scheduled requests,
TTFT includes requests with at least one token, request latency includes successful
finishes only, and inter-token latency contributes one sample per adjacent token
pair. Rate denominators begin at the earliest request creation time and require a
positive observation window. Snapshot allocation/sorting is intentionally outside
the per-iteration scheduling hot path.

## Model executor

The implemented M17 model and M18 generation baseline form a narrowly specified
CPU Float32 Llama-style decoder. It uses equal Q/K/V head counts, full
adjacent-pair RoPE over an even head dimension, pre-attention and pre-MLP RMSNorm,
bias-free projections, causal
attention, residuals, SwiGLU, final normalization, and an explicit untied LM head.
All projection weights use row-major `[input_features,output_features]` layout.
GQA/MQA, partial rotary dimensions, biases, tied embeddings, quantization, and
arbitrary checkpoint graphs are rejected rather than guessed.

```mermaid
flowchart LR
    Tokens[Token IDs T] --> Embed[Embedding V x H]
    Embed --> Block[Repeat L decoder blocks]
    Block --> ANorm[Attention RMSNorm]
    ANorm --> QKV[Q / K / V projections]
    QKV --> Rope[RoPE on Q and K]
    Rope --> Causal[Causal attention]
    Causal --> Residual1[O projection + residual]
    Residual1 --> FNorm[FFN RMSNorm]
    FNorm --> SwiGLU[Gate / up + SwiGLU]
    SwiGLU --> Residual2[Down projection + residual]
    Residual2 --> Block
    Block --> Final[Final RMSNorm + LM head]
    Final --> Logits[Logits T x V]
```

`CpuLlamaModel` moves a validated `LlamaWeights` bundle into one model owner. Its
const forward path allocates independent activations and has no shared mutable
state. Because Tensor storage is reference-counted, thread safety still requires
that a caller not mutate a shallow weight alias retained before construction. The
optional diagnostic path preserves post-layer Tensor handles and the final
normalized state; ordinary execution does not retain those intermediates.

`CpuLlamaGenerator` is the single-sequence session owner. It contains the model,
one `ContiguousKvCache`, and one sequence handle and is deliberately noncopyable,
nonmovable, and not thread-safe. The cache owns fixed-capacity
`[layer,head,capacity,dimension]` K/V storage; temporary attention tensors remain
ordinary CPU Tensor allocations. A caller cannot outlive or accidentally pair the
session's handle with another cache.

```mermaid
flowchart LR
    Prompt[Prompt token IDs] --> Prefill[Full causal prefill]
    Prefill --> PromptKV[Capture rotated K / V per layer]
    PromptKV --> Cache[(Contiguous KV cache)]
    Prefill --> Select[Greedy last-row selection]
    Select --> Token[Generated token]
    Token --> Decode[One-token decode at cache length]
    Cache --> Prefix[Copy history into attention prefix]
    Prefix --> Decode
    Decode --> NewKV[Transactional new-token K / V]
    NewKV --> Cache
    Decode --> Select
```

Prefill performs all model math before populating cache storage. Decode validates
that the supplied position equals the current cache length, copies each cached
layer prefix directly into a caller-owned buffer with one reserved tail row,
writes the current token's K/V into that tail, and computes attention. The model
returns the new K/V separately; the session appends it only after the entire token
step and final logits projection succeed. This keeps model failures from advancing
logical cache length. Multi-call execution assumes the caller does not mutate the
same sequence concurrently.

The `HELIXLM1` container is a deterministic little-endian format with a fixed
128-byte versioned header, fixed-size tensor directory, exact role/layer/shape
records, and 64-byte-aligned Float32 payloads. Loading validates file size,
configuration arithmetic, canonical nonoverlapping offsets, zero padding, finite
values, an FNV-1a metadata checksum, and one checksum per payload before returning
a model. FNV provides accidental-corruption detection, not authenticity. Details
and the exact byte layout are in `docs/MODEL_SUPPORT.md`.

The current CPU path performs full causal prefill and single-token contiguous-KV
decode and repacks token-major Q/K/V tensors around the existing head-major
attention API. Greedy selection, EOS/length/context stops, and a token-ID CLI are
implemented. Scheduler integration, batched/paged model decode, tokenizer
integration, public-checkpoint conversion, and CUDA model dispatch remain M19 or
later work. A model-specific conversion utility has intentionally not been guessed
because no public checkpoint compatibility is claimed yet.

## Runtime execution and asynchronous work

The following sequence is the M19 multi-request target, not a claim that the M18
single-sequence generator already drives the scheduler:

```mermaid
sequenceDiagram
    participant Client
    participant Scheduler
    participant Engine
    participant Model
    participant KV
    Client->>Scheduler: submit(request)
    loop inference iteration
        Scheduler->>Engine: build batch within budgets
        Engine->>Model: prefill/decode(batch)
        Model->>KV: read prior K/V and append new K/V
        Model-->>Engine: logits
        Engine->>Scheduler: generated tokens/completions
    end
    Scheduler-->>Client: completed response + metrics
```

CUDA execution starts with one explicit stream. Additional streams are introduced
only with event-defined dependencies. Pinned staging buffers support genuinely
asynchronous host-device copies. Each overlap experiment records both latency and
throughput because concurrency can increase contention.

## Observability and benchmark integrity

Logging has severity, source context, and a replaceable sink without imposing a
large dependency. Allocator and scheduler metrics are snapshots with documented
units. Profiling ranges may be added without changing execution semantics.

Every benchmark report records hardware, software, dimensions, precision, build
mode, warmups, measured iterations, timing mechanism, and synchronization. GPU
benchmarks use CUDA events for device work and may separately report end-to-end
host latency.

## Security and robustness boundaries

- Model files and CLI dimensions are untrusted inputs and receive bounds/overflow
  validation.
- Resource exhaustion returns explicit errors; it does not corrupt allocator or KV
  state.
- Allocations, streams, events, and library handles are released by RAII.
- CPU sanitizer jobs cover host memory and undefined behavior.
- CUDA runtime validation includes `compute-sanitizer` when hardware is available.

## Deliberate limitations

The initial architecture does not promise training, autograd, arbitrary model
graphs, arbitrary strides, production serving APIs, quantization, prefix sharing,
preemption, or multi-GPU inference. Each can be considered only after the core
runtime is correct and measured.
