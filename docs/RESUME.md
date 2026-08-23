# Resume Notes

Use only bullets whose validation context you can explain. The CUDA code in this
repository is compile-validated but has not run on NVIDIA hardware; do not turn
that into a GPU performance claim.

**Helix — High-Performance LLM Inference Runtime | C++20, CUDA, CMake**

- Engineered a CPU-first C++20/CUDA inference runtime with checked Tensor/storage
  ownership, deterministic reference kernels, causal attention, contiguous and
  paged KV caches, request scheduling, and continuous batching; maintained 148
  passing Debug, Release, and ASan/UBSan CPU tests.
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
- Integrated the scheduler, CPU Llama executor, and global paged KV pool into a
  synchronous multi-request engine with transactional failure/cancellation cleanup;
  optimized complete-prompt execution with bulk page append, reducing median CPU
  trace time by 15.8%–51.3% across three disclosed synthetic workloads.
- Composed the reference kernels into a narrow Float32 Llama-style full-prefill
  model and created a deterministic versioned weight container with checked
  tensor roles/shapes/offsets, 64-byte payload alignment, corruption checksums,
  finite-value validation, and transactional RAII loading.
- Built transactional single-sequence prefill/decode and greedy generation over a
  contiguous KV cache; matched cached logits/tokens against full-prefix
  recomputation and measured 3.69x–56.79x next-position CPU speedups across
  disclosed 8–64-token synthetic fixtures.

## Short project description

Built Helix to understand the low-level mechanisms beneath modern LLM inference
runtimes without delegating execution to PyTorch. The project emphasizes explicit
memory lifetime, correct asynchronous design, CPU/GPU oracle comparisons, honest
hardware-status labeling, and benchmark-controlled optimization.
