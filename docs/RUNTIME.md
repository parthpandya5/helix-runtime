# CPU Runtime Execution Engine

## Status and scope

Milestone 19 now has a synchronous CPU correctness engine. `CpuRuntimeEngine`
connects the implemented FIFO scheduler, `CpuLlamaModel`, one global
`PagedKvCache`, greedy token selection, request completion, and physical K/V
reclamation. It accepts multiple requests and performs genuine model work for the
mixed prefill/decode batches produced by continuous replacement.

This is not yet a CUDA execution engine. It has no device-resident block table,
CUDA stream, event dependency, allocator-backed Tensor storage, or fused model
batch. Scheduled work items execute serially on the calling thread using Float32
CPU reference kernels. Those boundaries are deliberate and are not hidden by the
`Runtime` name.

## Ownership

```mermaid
flowchart TD
    Client[Caller] -->|move Request| Engine[CpuRuntimeEngine]
    Engine --> Scheduler[FifoScheduler]
    Engine --> Model[CpuLlamaModel + weights]
    Engine --> Pool[One PagedKvCache]
    Engine --> Sessions[RequestId to Session map]
    Sessions --> Handle[Paged sequence handle]
    Sessions --> Logits[Next-token logits]
    Scheduler --> History[Terminal request history]
    Pool --> Blocks[(Physical CPU K/V blocks)]
```

The engine is the sole mutable owner of its scheduler, model, cache facade, and
session map. The paged cache owns two long-lived physical pool tensors. A session
owns only a cache-tagged sequence handle, optional EOS token, and logits needed to
select its next token. Terminal `Request` objects remain owned by the scheduler so
callers can inspect output and lifecycle metrics; their model sessions and K/V
blocks are released immediately.

`CpuRuntimeEngine` is noncopyable, nonmovable, and single-thread-owned. Returned
request references are invalidated only when the engine dies, but their generated-
token spans must not be retained across another iteration that can append.

## Submission contract

`submit` accepts a move-only waiting `Request` and an optional EOS token. Before
creating observable scheduler state it validates:

- every prompt/EOS token is within the model vocabulary;
- prompt length and generation budget fit the model context;
- the maximum physically cached prefix can fit in the global page pool;
- the request ID is not already owned by the engine.

For a prompt of `P` tokens and a length budget of `N`, at most
`P + max(N - 1, 0)` tokens need K/V storage. The final length-limited output is
selected from logits already available and is not fed back through the model.
Submission therefore does not reject an exact-boundary request merely to reserve
an unused final-token page.

The engine creates an empty paged sequence before moving the request into the
scheduler. If either session-map or scheduler insertion fails, it releases that
sequence. Invalid submissions leave the scheduler, session count, and physical
block pool unchanged.

## Iteration protocol

```mermaid
sequenceDiagram
    participant C as Caller
    participant E as CPU engine
    participant S as FIFO scheduler
    participant M as CPU model
    participant K as Paged KV cache
    C->>E: run_iteration(now)
    E->>K: snapshot actual free blocks
    E->>S: schedule_iteration(free blocks, now)
    S-->>E: ordered mixed batch
    loop each work item, serially
        E->>M: prefill range or decode one token
        M->>K: read validated K/V prefix
        E->>K: append successful new K/V
    end
    E->>S: complete_iteration(ordered results, now)
    S-->>E: terminal request IDs
    E->>K: release each terminal sequence
    E-->>C: work, completions, releases, and capacity snapshots
```

`run_iteration` is synchronous: scheduling, all model operations, logical
completion, and reclamation finish before it returns. At most one scheduler batch
exists, and no in-flight state escapes the call. The completion vector reserves
its maximum size before scheduling so its ordinary growth cannot strand a newly
opened scheduler iteration.

The cache's measured free-block count, rather than a separate mirrored counter,
is passed to the scheduler. Before every prefill or decode operation, the engine
checks that physical cache length equals the scheduled logical token offset. For
every nonterminal generated token, it eagerly evaluates that token and appends its
K/V before scheduler completion, preserving this equality for the next iteration.
The final length-limited token needs no append; the scheduler explicitly gives it
zero new-block demand, so it can finish at an exact block boundary with no free
page.

An EOS token is not predictable until logits are sampled. The scheduler therefore
reserves capacity conservatively for a non-final decode; if the result is EOS, the
engine skips the unused model step and reclaims the request's actual pages during
the same call.

## Prefill paths

Two implementations remain selectable through `CpuPrefillMode`:

- `kSequentialDecode` is the retained baseline. Every scheduled prompt token
  executes through one-token decode and appends one K/V record. It supports
  arbitrary scheduler chunk boundaries and makes the incremental semantics easy
  to audit.
- `kFullPrompt` is the default. When the scheduler delivers an entire new prompt,
  the model executes one full causal prefill, retains only the final logits row,
  and bulk-appends its layer-major `[heads,tokens,head_dimension]` K/V artifacts.
  Chunked or resumed prompts fall back to the sequential path.

`PagedKvCache::append_layer_major` validates every tensor and reserves the entire
resulting block table before ownership changes. It then copies maximal spans into
partial and whole physical blocks while holding the cache lock. Pool exhaustion,
bad shapes, zero token ranges, or metadata allocation failure leave sequence
length and block ownership unchanged. Its `append_count` continues to count
tokens, not API calls.

The paired Release benchmark retains both prefill modes and pins compact-copy
decode so only prompt execution changes. Full-prompt execution reduced median CPU
trace time by 51.3%, 36.0%, and 15.8% in three disclosed tiny workloads. The
decreasing benefit is expected because decode accounts for more of the larger
trace. Exact environments, shapes, timing boundaries, variation, and commands are
in `PERFORMANCE.md`.

## Paged decode paths

`CpuRuntimeConfig::paged_decode_mode` retains two implementations:

- `kCompactCopy` allocates `[heads,history+1,head_dimension]` key and value
  tensors per layer, copies every cached page prefix into them, writes current K/V
  into the final row, then calls contiguous attention.
- `kDirectBlockTable` is the measured default. It acquires a move-only layer read
  lease and computes attention over cached physical spans plus the current
  uncommitted K/V tail. The model returns that tail for one all-layer append only
  after complete token execution succeeds.

The direct path preserves physical sequence length throughout model execution and
checks the leased length against the scheduled position under the cache lock. It
supports an empty history, which is required by sequential prefill at position
zero. The lease is released before the engine appends; attempting to append while
holding it would deadlock on the cache's nonrecursive mutex.

In paired isolated layer calls, direct traversal reduced median CPU time by
13.3%–27.7% and removed 2 KiB–1 MiB of combined-K/V temporaries. Complete runtime
trace medians fell by 3.97%, 6.23%, and 0.08%; the largest case is effectively a
tie relative to measured variation. Both modes remain executable so this tradeoff
can be remeasured on different workloads rather than encoded as an unsupported
universal claim.

## Completion, failure, and cancellation

A successful prefill carries no output token. Decode selects the lowest-ID finite
maximum from the session's last logits and reports exactly one token plus an EOS
flag. Model/cache exceptions are converted into ordered failed work completions;
the scheduler transitions that request to `failed`, retains contextual error text,
and the engine releases its physical sequence.

Completion validates the whole ordered vector before logical request mutation.
After it returns, the engine releases every reported terminal session before
moving the retirement records into the public result. This order is intentional:
moving the vector first would make reclamation iterate an empty, moved-from
container.

`cancel` delegates state validation to the scheduler, then immediately releases
the corresponding empty, prefill, or decode sequence. Cancellation is rejected
while a scheduler iteration is in flight, matching the single-iteration ownership
contract.

## Observability

Each iteration result reports:

- iteration identity and exact scheduled work;
- ordered work completions and terminal requests;
- actual free physical blocks before and after execution;
- scheduled prefill/decode token counts;
- token-budget and K/V-capacity limiting flags.

`CpuRuntimeStats` combines scheduler and paged-cache snapshots with active session,
executed iteration, failed work, full-prompt prefill, and sequential-prefill token
counters. Scheduler lifecycle metrics remain available at an explicit monotonic
observation time.

## Correctness coverage

Nine engine tests exercise:

- chunked prefill and multi-step paged decode against full-prefix recomputation;
- full-prompt prefill and bulk K/V append across page boundaries;
- compact-copy and direct block-table decode producing identical request output;
- real-model continuous replacement with mixed prefill/decode work;
- exact-boundary physical exhaustion, reclamation, and block reuse;
- EOS, running cancellation, waiting cancellation, and complete reclamation;
- 32 mixed-length requests under constrained-pool pressure, with every generated
  sequence checked against independent full recomputation and final leak checks;
- invalid configuration/submission with no leaked sessions;
- injected non-finite model output converted to request failure and reclaimed K/V.

The bulk cache, scheduler final-token rule, and paged model decode have additional
focused tests. The full Debug, Release, and ASan/UBSan CPU suites pass 156/156. A
CUDA 12.6 warnings-as-errors build passes the same 156 host tests and registers 57
GPU tests, all skipped without an NVIDIA driver/device.

## Current limitations

- Work items execute serially; this engine does not form vectorized model batches.
- Full-prompt prefill is used only for a complete first prompt chunk. Partial
  prefill remains the slower one-token baseline.
- Direct one-token attention holds a cache-wide read lease for each layer, so it
  serializes unrelated mutation in the same physical pool.
- The physical K/V pool is CPU Tensor storage and does not use `HelixAllocator`.
- There are no CUDA streams, events, asynchronous copies, or device execution.
- The strict FIFO policy has no preemption, priority, prefix sharing, or terminal
  request archive/reaping API.
- Host allocation failure while constructing an error message is not a recoverable
  fault boundary; ordinary model/cache failures are handled and tested.

The next execution-engine step is a CUDA model path with device-resident paged K/V
and explicit stream ownership. Stream overlap is considered only after that path
is correct and measured on NVIDIA hardware.
