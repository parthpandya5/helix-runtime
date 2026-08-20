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

### M3 — CUDA runtime layer — In progress; compilation/runtime hardware pending

- checked CUDA errors with operation context
- move-only device guard, stream, and event wrappers
- synchronous and asynchronous copy primitives
- initial direct device allocation/storage
- runtime capability reporting and hardware-gated tests

Exit criteria include leak-free resource tests on real NVIDIA hardware. Compile-only
validation is tracked separately.

### M4 — Basic CUDA kernels — Planned; hardware pending

- manual vector add, multiply, ReLU, and SiLU
- CPU-oracle comparisons across non-block-aligned sizes
- event-timed CPU/CUDA comparisons with warmups
- documentation of launch-overhead crossover behavior from measured data

### M5 — Reduction and softmax — Planned; hardware pending

- naive, shared-memory, and warp-assisted reduction variants
- correct row-wise softmax followed by evidence-driven optimization
- latency, throughput, and effective-bandwidth reporting where meaningful
- documented thread layout, synchronization, and numerical behavior

### M6 — Matrix multiplication — Planned; hardware pending

- one-output-element-per-thread naive SGEMM
- shared-memory tiled SGEMM supporting edge tiles
- further tiling/vectorization only after profiling
- honest cuBLAS SGEMM comparison when available
- FLOP count, traffic model, arithmetic-intensity, and occupancy analysis

## Phase C: device memory systems

### M7 — Caching GPU allocator — Planned; hardware pending

- aligned best-fit baseline with segment/block metadata
- split and same-segment adjacent merge
- reusable cache and explicit release policy
- reserved/allocated/free/peak/counter statistics
- specified external and internal fragmentation metrics
- native CUDA versus Helix allocation-trace benchmarks

Workloads include fixed, randomized, small-heavy, large-heavy, bursty, and
LLM-shaped traces. Strategy conclusions depend on measurements.

### M8 — Stream-aware allocation — Planned; hardware pending

- documented immediate-reuse race
- event-backed deferred free/reclamation
- explicit stream-recording contract
- real-hardware test that would detect unsafe reuse
- comparison with synchronized baseline and `cudaMallocAsync` where supported

## Phase D: transformer primitives and memory

### M9 — Transformer kernels — Planned

- CPU and CUDA RMSNorm, RoPE, SwiGLU, and projection paths
- custom educational GEMM remains selectable
- cuBLAS production path is separately named and documented if added

### M10 — Contiguous attention baseline — Planned

- scaled QK-transpose, causal mask, softmax, probability-times-V stages
- CPU reference and explicit temporary-memory accounting
- CUDA correctness validation before fusion work

### M11 — Contiguous KV cache — Planned

- layer-aware K/V shape and capacity model
- create, append, retrieve, and release sequence operations
- tests showing decode reuses prior K/V
- memory formulas checked against allocated bytes

### M12 — Paged KV cache — Planned

- fixed-token physical blocks and strongly typed block/sequence IDs
- logical-to-physical block tables and free-block pool
- transactional append on block exhaustion
- deterministic allocation/reclamation tests
- fragmentation/capacity comparison with contiguous sequence allocation

Reference counting is deferred until prefix sharing creates a real need.

### M13 — Paged attention — Planned; GPU optimization hardware pending

- correctness path reading K/V through sequence block tables
- comparison with contiguous attention
- memory-access and coalescing analysis
- measured latency, overhead, fragmentation, and capacity on target hardware

## Phase E: requests and continuous execution

### M14 — Request model — Planned

- stable request/token IDs
- validated waiting, prefill, decoding, terminal state transitions
- generation limits, cancellation, failure context, and timestamps
- exhaustive state-transition tests

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
