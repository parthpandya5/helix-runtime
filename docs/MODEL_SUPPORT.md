# Helix Model Support

## Status

Milestones 17–19 implement one narrow, deterministic Llama-style model,
single-sequence generation session, and multi-request CPU orchestration path. They
are Float32 correctness baselines for full causal prefill and contiguous or paged
KV decode, not a claim
of compatibility with every model whose configuration contains the word “Llama.”
The implementation, native weight container, unit tests, sanitizer suite, and CPU
benchmarks are complete. CUDA model execution, fused tensor batching, tokenizer
integration, and real checkpoint conversion remain later gates.

## Exact architecture contract

For vocabulary `V`, hidden width `H`, intermediate width `I`, layer count `L`,
attention heads `A`, head dimension `D = H/A`, and prompt length `T`, Helix accepts:

- positive dimensions, `H % A == 0`, and an even `D`;
- one Q, K, and V head per attention head; grouped-query and multi-query attention
  are not supported;
- Float32 CPU weights and activations in contiguous row-major storage;
- bias-free embedding/projection/MLP operations;
- pre-attention and pre-MLP RMSNorm;
- full adjacent-pair RoPE across each head dimension;
- causal scaled dot-product attention;
- a SwiGLU MLP, residual connections, final RMSNorm, and an explicit untied LM
  head.

The complete baseline is:

```text
token_ids[T]
  -> embedding[V,H]                         => hidden[T,H]
  -> repeat L times:
       RMSNorm
       Q/K/V linear[H,H]
       reshape[T,A,D] + RoPE(Q,K)
       causal attention[A,T,D]
       output linear[H,H] + residual
       RMSNorm
       gate linear[H,I], up linear[H,I]
       SiLU(gate) * up
       down linear[I,H] + residual
  -> final RMSNorm
  -> LM head[H,V]                           => logits[T,V]
```

Every linear weight uses `[input_features, output_features]` layout. Attention
temporarily repacks token-major `[T,A,D]` projections into the existing
head-major `[A,T,D]` attention contract, then repacks its result back to `[T,H]`.
That copying is intentionally visible in this first baseline.

`CpuLlamaModel::forward` returns logits for every prompt position. The opt-in
`forward_with_trace` additionally retains each post-layer hidden state and the
final normalized state. Tests compare all of those stages with an independent
`std::vector<float>` oracle; the ordinary forward path does not retain them.

## Prefill, cached decode, and generation

`CpuLlamaModel::prefill_with_kv` runs the same full causal path as `forward` and
additionally returns one rotated key tensor and one value tensor per layer, each
head-major `[A,T,D]`. `CpuLlamaModel::decode_one` consumes one token at absolute
position `P`, requires an existing cache length of exactly `P`, applies RoPE at
that absolute position, and attends the single query over `P` cached tokens plus
its current K/V. It returns `[1,V]` logits and the new token's all-layer K/V shaped
`[L,A,D]` without changing the cache.

`CpuLlamaGenerator` owns the model, a `ContiguousKvCache` sized to the model's
maximum sequence length, and one sequence handle. It is noncopyable, nonmovable,
single-sequence, and not thread-safe. Prefill computes every model artifact before
appending prompt tokens. Decode computes the complete token result before append.
Therefore ordinary validation/model failures leave cache length unchanged. Cache
append itself is a prevalidated, fixed-capacity copy with no allocation after the
model step.

The first decode baseline retrieved an owning compact K/V snapshot per layer and
then copied that snapshot into a second buffer with a current-token tail.
`copy_layer_prefix_out` now copies the locked history directly into that final
`[A,P+1,D]` buffer, validates the caller's expected length under the cache mutex,
and leaves the tail untouched for the current K/V. This preserves cache ownership
while removing two Tensor temporaries and one history-copy pass per layer.

The same `decode_one` algorithm is instantiated for `PagedKvCache`: both cache
types expose checked length, geometry, and prefix-copy contracts, while the model
never owns or mutates either cache. `CpuRuntimeEngine` uses this overload for one
paged sequence per live request and appends returned all-layer K/V only after a
successful model step. A deterministic test crosses a paged block boundary and
compares every sequential decode position with full-prefix recomputation.

When a new prompt is scheduled as one complete range, the M19 engine calls
`prefill_with_kv` once, retains only the final `[1,V]` logits row, and passes the
layer-major K/V artifacts to `PagedKvCache::append_layer_major`. Partial prompt
ranges retain token-at-a-time decode so arbitrary scheduler chunking remains
correct. These are selectable benchmark paths rather than an erased baseline.

`select_greedy_token` examines the last logits row, requires finite Float32 CPU
values, and selects the lowest token ID among equal maxima. `generate` emits until
the requested count, optional EOS token, or context capacity prevents another
decode. A selected token is appended to K/V only when it becomes the input to a
later decode step, so a final emitted token may not be cached. `GenerationResult`
reports the actual cache count instead of implying otherwise.

The `helix_generate` example loads `HELIXLM1` and accepts prompt token IDs:

```bash
./build/examples/helix_generate \
  --model ./models/example.helix \
  --max-new-tokens 16 \
  1 42 17
```

This explicit token-ID boundary avoids making a tokenizer or Python a runtime
dependency. It is not text-in/text-out model support.

## Weight tensors

The runtime requires exactly `3 + 9*L` tensors in this order:

| Scope | Role | Shape |
|---|---|---:|
| Global | token embedding | `[V,H]` |
| Each layer | attention norm | `[H]` |
| Each layer | query, key, value | each `[H,H]` |
| Each layer | attention output | `[H,H]` |
| Each layer | FFN norm | `[H]` |
| Each layer | gate, up | each `[H,I]` |
| Each layer | down | `[I,H]` |
| Global | final norm | `[H]` |
| Global | LM head | `[H,V]` |

`LlamaWeights` is move-only and validates every tensor's device, dtype,
contiguity, and exact shape. `CpuLlamaModel` moves the weight bundle into one
owner. Tensor storage itself is reference-counted, so a caller that retained a
shallow alias before model construction could still mutate the same bytes. Const
model APIs never mutate weights, and concurrent forward calls are safe only while
no external alias mutates backing storage. A future immutable weight-storage type
would enforce this boundary rather than merely documenting it.

## Native model file v1

`write_llama_model_file` and `read_llama_model_file` implement a deterministic,
little-endian container. Streams must be seekable and start at offset zero. The
runtime does not require Python to read it.

### Header

The v1 header is exactly 128 bytes:

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | magic `HELIXLM1` |
| 8 | 4 | format version (`1`) |
| 12 | 4 | endian marker (`0x01020304`) |
| 16 | 4 | header size (`128`) |
| 20 | 4 | model kind (`1`, Llama-style) |
| 24 | 8 | exact file size |
| 32 | 8 | tensor-directory offset (`128`) |
| 40 | 8 | tensor count |
| 48 | 8 | directory-entry size (`80`) |
| 56 | 8 | first payload-region offset |
| 64 | 48 | `V`, `H`, `I`, `L`, `A`, maximum sequence length as six `uint64` values |
| 112 | 4 | RMSNorm epsilon as IEEE-754 Float32 bits |
| 116 | 4 | RoPE theta as IEEE-754 Float32 bits |
| 120 | 8 | FNV-1a checksum of the header with this field zeroed plus the directory |

### Directory and payloads

Each 80-byte directory entry contains a stable tensor-role code, layer index,
dtype code, rank, four dimension slots, payload offset, byte length, alignment,
and payload checksum. Unused dimension slots must be zero. Global tensors use a
sentinel layer index. Payloads contain little-endian IEEE-754 Float32 bits, begin
at 64-byte-aligned offsets, and have zero-filled alignment gaps.

The loader checks, before exposing a model:

- magic, version, endian marker, model kind, header sizes, and exact stream size;
- all configuration invariants and checked host-size conversions;
- the exact `3 + 9*L` tensor count and canonical directory size/data offset;
- metadata checksum before trusting directory records;
- canonical tensor order, role/layer identity, dtype, rank, shape, byte length,
  alignment, nonoverlapping offsets, and complete file coverage;
- zero alignment padding, every payload checksum, and finite Float32 values;
- the assembled `LlamaWeights` contract once more before return.

Payload tensors are materialized into local RAII objects while checksums are
verified. An error destroys those locals; a partially loaded `LoadedLlamaModel` is
never returned. FNV-1a detects accidental corruption but is not cryptographic
authentication. Treat model files as untrusted structured input, not as signed
artifacts.

## Correctness and validation

The deterministic model test uses two layers, two heads, odd prompt length three,
and independently implements embedding lookup, RMSNorm, every projection, RoPE,
causal attention, residuals, SwiGLU, final normalization, and the LM head with
plain vectors. It compares each layer state, the final normalized state, and every
logit within `5e-5`, then checks bitwise repeatability. Boundary tests cover empty,
overlength, maximum-length, and out-of-vocabulary token sequences, invalid head
geometry, non-finite configuration values, wrong layer counts, dtype errors, and
bad weight shapes.

Generation tests compare prompt logits and two successive cached decode positions
with full-prefix recomputation, then compare a complete four-token greedy sequence
with repeated full recomputation. They cover decode at position zero, deterministic
tie-breaking, non-finite logits, invalid EOS/token/cache geometry, zero budget,
EOS, context exhaustion, repeated-prefill misuse, and transactional cache length
on failures. The direct prefix-copy test verifies multi-head layout, untouched tail
rows, stale expected-length rejection, insufficient output capacity, and alias
rejection.

Model-file tests cover bit-exact deterministic round trips, executable loaded
weights, filesystem APIs, truncation, trailing bytes, wrong magic/version/endian
markers, metadata and payload corruption, nonzero padding, structurally invalid
but checksum-consistent ranks/offsets, invalid weights, non-finite weights, and
nonzero stream origins.

Validation on 2026-08-23 through M19 implementation commit `8086257` and benchmark
methodology commit `2a6b365`:

- normal, Release, and Apple Clang ASan/UBSan CPU suites: 148/148 tests passed;
- relevant Release tests and benchmark smoke workloads passed;
- CUDA 12.6.85/GCC 13.2 warnings-as-errors build: all targets compiled and all
  148 hardware-independent tests passed;
- 57 GPU runtime tests remained explicitly skipped because the compile container
  has no NVIDIA driver or device.

## Measured CPU baselines

Release-mode full-prefill and checked in-memory-load measurements are recorded in
[`PERFORMANCE.md`](PERFORMANCE.md). `forward` includes its temporary tensor
allocations and full model math; model construction and weight initialization are
outside timing. The load fixture includes stream construction, metadata and
payload validation/checksums, endian decoding, Tensor allocation, and final weight
validation, but excludes filesystem I/O and serialization.

These measurements establish the first implementation. They do not show an
optimization, GPU behavior, production inference throughput, or comparison with
BLAS/framework runtimes.

The M18 next-position fixture compares one cached decode against recomputing the
same complete prefix. Model/cache setup and prompt prefill are excluded; the cached
timed region includes one-token math, cache history reads, final cache append, and
temporary allocations. The paired cache-staging fixture retains the original
two-stage snapshot algorithm beside the direct-prefix path and includes output
allocation and current-tail copies for both. Exact medians, variation, model
shapes, repetitions, and commands are in `PERFORMANCE.md`.

The M19 paired engine trace holds scheduler/decode/pool behavior constant and
compares sequential-token prompt execution with full-prompt execution plus bulk
paged append. The default reduced median CPU trace time by 15.8%–51.3% across the
three tiny disclosed workloads. It measures serial CPU orchestration, not fused
batch or GPU throughput.

## Deliberate limitations

- CPU Float32 only for complete model execution.
- Complete new prompts use one full prefill; chunked runtime prefill falls back to
  sequential one-token execution rather than a multi-token continuation kernel.
- Cached decode uses a fixed-capacity contiguous CPU cache and materializes one
  `[A,P+1,D]` K/V pair per layer for the current attention call.
- The standalone generator supports one sequence and greedy selection only. The
  runtime supports several paged sessions but executes work items serially and has
  no sampling or fused tensor batch.
- No tokenizer and no supported public checkpoint conversion tool yet.
- No GQA/MQA, tied-embedding option, biases, partial rotary dimensions, sliding
  windows, mixture-of-experts blocks, quantization, or arbitrary model graphs.
- No memory mapping or zero-copy file loading; each payload is decoded and copied
  into owned aligned Tensor storage.

Those omissions are compatibility boundaries, not silently accepted variants.
