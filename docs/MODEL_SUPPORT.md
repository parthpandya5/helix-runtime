# Helix Model Support

## Status

Milestone 17 implements one narrow, deterministic Llama-style model in the CPU
backend. It is a Float32 correctness baseline for full causal prefill, not a claim
of compatibility with every model whose configuration contains the word “Llama.”
The implementation, native weight container, unit tests, sanitizer suite, and CPU
benchmarks are complete. CUDA model execution, KV-cached decode, tokenizer
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

Model-file tests cover bit-exact deterministic round trips, executable loaded
weights, filesystem APIs, truncation, trailing bytes, wrong magic/version/endian
markers, metadata and payload corruption, nonzero padding, structurally invalid
but checksum-consistent ranks/offsets, invalid weights, non-finite weights, and
nonzero stream origins.

Validation on 2026-08-23 for implementation commit `987c3b2`:

- normal and Release CPU suites: 129/129 tests passed;
- Apple Clang ASan/UBSan suite: 129/129 tests passed;
- CUDA 12.6.85/GCC 13.2 warnings-as-errors build: all targets compiled and all
  129 hardware-independent tests passed;
- 57 GPU runtime tests remained explicitly skipped because the compile container
  has no NVIDIA driver or device.

## Measured CPU baseline

Release-mode full-prefill and checked in-memory-load measurements are recorded in
[`PERFORMANCE.md`](PERFORMANCE.md). `forward` includes its temporary tensor
allocations and full model math; model construction and weight initialization are
outside timing. The load fixture includes stream construction, metadata and
payload validation/checksums, endian decoding, Tensor allocation, and final weight
validation, but excludes filesystem I/O and serialization.

These measurements establish the first implementation. They do not show an
optimization, GPU behavior, production inference throughput, or comparison with
BLAS/framework runtimes.

## Deliberate limitations

- CPU Float32 only for complete model execution.
- Full prompt prefill recomputes the whole causal history on every call.
- No KV-cached decode, token selection, generation loop, batching, or runtime
  scheduler integration yet.
- No tokenizer and no supported public checkpoint conversion tool yet.
- No GQA/MQA, tied-embedding option, biases, partial rotary dimensions, sliding
  windows, mixture-of-experts blocks, quantization, or arbitrary model graphs.
- No memory mapping or zero-copy file loading; each payload is decoded and copied
  into owned aligned Tensor storage.

Those omissions are compatibility boundaries, not silently accepted variants.
