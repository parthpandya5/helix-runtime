# KV Cache

## Status

The M11 contiguous and M12 paged Float32 CPU caches are implemented,
CPU-validated, sanitized, and benchmarked. They share append/retrieve/release
semantics while exposing the memory-capacity and traversal tradeoffs between a
per-sequence allocation and a global fixed-block pool. M13 paged attention remains
planned; there is not yet a GPU-resident KV-cache path.

## Why cache keys and values?

In autoregressive decode, a request adds one token while continuing to attend to
all earlier tokens. The earlier tokens' projected keys and values do not change.
Recomputing those projections on every decode step repeats work and requires the
old hidden states. A KV cache stores each layer's projected K/V vectors once and
appends only the new token.

For Float32 with `L` layers, `H` KV heads, and head dimension `D`, Helix stores per
token:

```text
bytes_per_token = L * 2 * H * D * sizeof(float)
                          ^ keys + values
```

Every multiplication is overflow-checked before storage is allocated or pointer
arithmetic is performed.

## Contiguous baseline

`ContiguousKvCache` owns a map of live sequences. Creating a sequence reserves its
entire declared token capacity in two CPU tensors:

```text
keys   [layers, kv_heads, capacity_tokens, head_dimension]
values [layers, kv_heads, capacity_tokens, head_dimension]
```

An append accepts one complete token shaped
`[layers, kv_heads, head_dimension]`. It copies each layer/head vector into the
current logical position and advances length only after every copy completes.
Retrieval returns an owning, compact layer snapshot shaped
`[kv_heads, current_length, head_dimension]`, which can be passed directly to the
M10 attention API. `copy_layer_out` uses caller-owned outputs to avoid allocation
in repeated retrieval and benchmark paths.

```mermaid
flowchart LR
    Token[Token K/V<br/>L x H x D] --> Append[append]
    Append --> Sequence[Sequence storage<br/>L x H x Capacity x D]
    Sequence --> Select[select layer]
    Select --> Compact[Compact history<br/>H x Length x D]
    Compact --> Attention[Contiguous attention]
```

The capacity stride means each head's active prefix is contiguous. Retrieving one
layer performs two `memcpy` operations per head—one for keys and one for values.
Unused capacity is uninitialized and is never exposed or read.

## Ownership and handles

The cache exclusively owns all sequence storage. `SequenceHandle` contains both a
process-unique cache identity and a monotonically increasing sequence identity.
This prevents an equal local sequence number from another cache instance from
releasing or mutating storage. Handles are non-owning capabilities: releasing a
sequence invalidates every copy, and subsequent use produces a contextual error.

Retrieved `KvLayer` tensors own independent compact storage, so their lifetime is
not tied to the sequence. This deliberate baseline copy is safe and makes the
attention boundary explicit; M13 will replace it with block-table traversal rather
than claiming that this path is zero-copy.

## Capacity, exhaustion, and accounting

Capacity is reserved transactionally at sequence creation. If
`capacity_tokens * bytes_per_token` overflows or would exceed `maximum_bytes`, no
sequence or counter is changed. Appending to a full sequence likewise preserves
length and statistics. The first pressure policy is rejection with
`kResourceExhausted`; there is no implicit eviction or preemption.

Statistics expose:

- currently allocated and peak bytes,
- live sequence count,
- active capacity tokens and written tokens,
- successful create, append, and release counts.

The contiguous baseline's sequence-level utilization is:

```text
active_tokens / active_capacity_tokens
```

and its unused reserved payload is:

```text
(active_capacity_tokens - active_tokens) * bytes_per_token
```

This is capacity slack, not the `HelixAllocator` block-fragmentation metric. A
request must choose one fixed maximum up front, and its unused tail cannot serve a
different sequence. That is the allocation problem M12 paging addresses.

## Thread safety

Each cache instance serializes public operations that access sequence state with
one mutex. Immutable configuration and byte formulas can be read without locking.
The coarse lock intentionally covers metadata updates and copies in these
correctness baselines; it gives a clear shared-state boundary and is tested with
concurrent appends to independent sequences. Fine-grained locking is deferred
until a measured scheduler workload demonstrates contention.

## Contiguous correctness coverage

Deterministic tests cover:

- multi-layer, multi-head layout and exact head-major retrieval,
- append-to-attention integration for single-token decode,
- full-sequence and whole-cache exhaustion without partial mutation,
- exact byte/capacity/statistic accounting and reclamation,
- empty sequences, invalid configurations, shapes, dtypes, layers, and outputs,
- empty, foreign, released, and cross-cache handles,
- output alias rejection for nonempty copies,
- concurrent appends to independent sequences,
- normal and ASan/UBSan execution.

Actual CPU workload baselines and their limitations are recorded in
`docs/PERFORMANCE.md`.

## Paged block cache

`PagedKvCache` allocates two long-lived, physical-major CPU tensors when the cache
is constructed:

```text
keys   [physical_blocks, layers, kv_heads, block_tokens, head_dimension]
values [physical_blocks, layers, kv_heads, block_tokens, head_dimension]
```

It assigns a fixed-token `PhysicalBlockId` only when an append crosses a logical
block boundary. A sequence's `std::vector<PhysicalBlockId>` block table maps a
logical position to pool storage:

```mermaid
flowchart LR
    Position[logical token position] --> Split[block index + token offset]
    Split --> Table[sequence block table]
    Table --> Id[physical BlockId]
    Id --> Pool[(K/V block pool)]
    Split --> Pool
```

For position `p`, `logical_block = p / block_tokens` and
`token_offset = p % block_tokens`. The physical-major layout keeps every physical
block's complete layer/head K/V payload together. Within one layer and head, a
block's token vectors are contiguous. Retrieving a compact layer therefore issues
two copies per head per logical block instead of two copies per head in the
contiguous cache.

### Free pool and ownership

The free pool is a fixed-capacity circular FIFO initialized with block IDs in
ascending order. It allocates no metadata during release, returns blocks in a
deterministic order, and prefers never-used blocks before recycled blocks. Parallel
arrays record the owning sequence, free membership, and whether each block has ever
been assigned. `PagedSequenceHandle` carries a cache identity and monotonic
sequence identity independently from the contiguous handle type.

When an append needs a new block, Helix validates the input and checks availability
before changing the free queue. It first grows the sequence table; only after that
potentially throwing allocation succeeds does it dequeue and assign the physical
block. All remaining writes are bounds-derived nonthrowing copies, so exhaustion or
host metadata OOM cannot leave a partial mapping. An append inside an already-owned
partial block remains legal even when the free pool is empty.

Release validates every block owner before mutation, then returns all blocks
without allocating. Reused storage is not cleared, but retrieval exposes only the
new sequence's written logical length; every visible token position is overwritten
before it can be read. There is one owner per block. Reference counting is
intentionally absent until prefix sharing creates shared ownership.

### Accounting and invariants

The pool reserves:

```text
pool_bytes = physical_blocks * block_tokens * bytes_per_token
```

Assigned capacity is `used_blocks * block_tokens`; per-sequence tail slack is less
than `block_tokens`. Statistics report reserved/used/free/peak bytes, total/used/
free blocks, active sequences/tokens, allocated and unused token slots, successful
lifecycle counts, block assignments, and reuse. Debug builds re-derive ownership,
block-table lengths, free membership, byte totals, token slots, and all major
counters after every mutation.

The deterministic comparison test uses two requests declared with contiguous
capacities of ten tokens but active lengths of one and five. The contiguous cache
assigns 20 token slots (160 bytes) and leaves 14 unused; a four-token paged pool
assigns 12 slots (96 bytes) and leaves six unused. This result is specific to an
over-reserved trace. If every contiguous request knew its exact final length,
paging could use more memory because each sequence retains up to one partial-block
tail.

### Paged correctness coverage

Seven deterministic tests cover:

- multi-layer/head values spanning odd and complete block boundaries,
- visible block-table order and compact retrieval,
- cached decode passed through the M10 attention oracle,
- transactional exhaustion and continued writes to a partial final block,
- FIFO reclamation, reuse, stale-data invisibility, and double-release rejection,
- invalid/overflowing configuration, handles, shapes, dtypes, layers, and outputs,
- exact statistics and contiguous-versus-paged tail accounting,
- concurrent append followed by complete reclamation,
- normal and ASan/UBSan execution.

Release-mode lifecycle and layer-copy comparisons are recorded in
`docs/PERFORMANCE.md`. The large-history layer-copy results are deliberately
unfavorable to paging because materialization must traverse many discontiguous
blocks; M13 will read block tables directly instead of treating compaction as the
final attention design.

## Current limitations

- Float32 CPU tensors only; no GPU-resident cache or asynchronous copy contract.
- Keys and values use the same head dimension.
- Retrieval compacts/copies a layer; attention does not yet consume block tables
  directly.
- Contiguous sequence capacity cannot grow; the paged pool cannot grow after
  construction.
- No prefix sharing, eviction, preemption, or reference counting yet.
- One mutex serializes all sequences.

The implementation is original. Fixed-size blocks, block tables, and KV reuse are
standard inference-runtime concepts; Helix keeps its own layout, ownership,
failure, and validation contracts visible rather than copying another runtime's
allocator or cache implementation.
