# Resume Notes

Use only bullets whose validation context you can explain. The CUDA code in this
repository is compile-validated but has not run on NVIDIA hardware; do not turn
that into a GPU performance claim.

**Helix — High-Performance LLM Inference Runtime | C++20, CUDA, CMake**

- Engineered a CPU-first C++20/CUDA inference runtime with checked Tensor/storage
  ownership, deterministic reference kernels, causal attention, contiguous and
  paged KV caches, request scheduling, and continuous batching; maintained 129
  passing normal, Release, and ASan/UBSan CPU tests.
- Implemented manual CUDA elementwise, reduction, stable-softmax, naive/tiled
  SGEMM, RMSNorm, RoPE, SwiGLU, and attention paths with RAII streams/events and
  hardware-gated CPU-oracle tests; compile-validated all CUDA targets with NVCC
  12.6 while explicitly deferring runtime/performance claims until GPU access.
- Designed a best-fit caching allocator with aligned split/coalesce, reservation
  limits, generation-tagged stale/double-free protection, internal/external
  fragmentation metrics, and event-backed deferred reuse for stream-ordered CUDA
  lifetimes.
- Built a paged KV manager with fixed physical blocks, logical block tables,
  deterministic FIFO reclamation, transactional exhaustion, cache-tagged handles,
  and copy-free read leases; optimized CPU attention value traversal by 2.36x–
  2.99x, then measured the remaining paged-layout overhead at 2.1%–6.7% against an
  equally optimized contiguous baseline.
- Implemented a validated FIFO token-budget scheduler with explicit request state
  transitions, KV-block accounting, continuous slot refill, and wait/TTFT/p50/p95/
  inter-token metrics, separating measured host-policy cost from synthetic
  utilization models.
- Composed the reference kernels into a narrow Float32 Llama-style full-prefill
  model and created a deterministic versioned weight container with checked
  tensor roles/shapes/offsets, 64-byte payload alignment, corruption checksums,
  finite-value validation, and transactional RAII loading.

## Short project description

Built Helix to understand the low-level mechanisms beneath modern LLM inference
runtimes without delegating execution to PyTorch. The project emphasizes explicit
memory lifetime, correct asynchronous design, CPU/GPU oracle comparisons, honest
hardware-status labeling, and benchmark-controlled optimization.
