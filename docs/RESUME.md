# Resume Notes

Use only bullets whose validation context you can explain. The CUDA code in this
repository is compile-validated but has not run on NVIDIA hardware; do not turn
that into a GPU performance claim.

**Helix — High-Performance LLM Inference Runtime | C++20, CUDA, CMake**

- Engineered a CPU-first C++20/CUDA inference runtime with checked Tensor/storage
  ownership, deterministic reference kernels, causal attention, contiguous and
  paged KV caches, request scheduling, and continuous batching; maintained 196
  passing Debug, Release, and ASan/UBSan CPU tests.
- Implemented manual CUDA elementwise, reduction, stable-softmax, naive/tiled
  SGEMM, embedding, layout, RMSNorm, RoPE, SwiGLU, and attention paths with RAII
  streams/events; composed them into a synchronous full-prefill model with owned
  device weights and explicit temporary-lifetime barriers, then compile-validated
  all CUDA targets and CPU-oracle tests with NVCC 12.6 while explicitly deferring
  runtime/performance claims until GPU access.
- Designed a best-fit caching allocator with aligned split/coalesce, reservation
  limits, generation-tagged stale/double-free protection, internal/external
  fragmentation metrics, and event-backed deferred reuse for stream-ordered CUDA
  lifetimes; optimized block and active-allocation metadata to reduce median cache
  trace CPU time by 53.1%–66.7% across five workloads while retaining measured
  small-allocation regressions as an explicit limitation.
- Built a paged KV manager with fixed physical blocks, logical block tables,
  deterministic FIFO reclamation, transactional exhaustion, cache-tagged handles,
  and copy-free read leases; optimized CPU attention value traversal by 2.36x–
  2.99x, then measured the remaining paged-layout overhead at 2.1%–6.7% against an
  equally optimized contiguous baseline.
- Added reference-counted, block-aligned KV prefix forks with immutable shared
  pages, divergence-safe exclusive tails, and logical-versus-resident accounting;
  measured 11.02x–320.67x lower child lifecycle time than copying 32 KiB–1 MiB
  prefixes while assigning no additional payload pages.
- Designed a bounded exact-token prefix cache with collision-verified hash lookup,
  persistent paged anchors, LRU entry/page budgets, and pressure eviction; integrated
  suffix-only scheduler/runtime execution and measured a candid 68.7% short-prompt
  regression followed by 1.80x and 6.04x speedups on disclosed 16/32-token CPU
  traces.
- Implemented a validated FIFO token-budget scheduler with explicit request state
  transitions, KV-block accounting, continuous slot refill, and wait/TTFT/p50/p95/
  inter-token metrics, separating measured host-policy cost from synthetic
  utilization models.
- Integrated the scheduler, CPU Llama executor, and global paged KV pool into a
  synchronous multi-request engine with transactional failure/cancellation cleanup;
  optimized complete-prompt execution with bulk page append, reducing median CPU
  trace time by 15.8%–51.3% across three disclosed synthetic workloads.
- Removed paged-decode history compaction with block-table attention over an
  uncommitted K/V tail, eliminating 2 KiB–1 MiB of temporary payload per measured
  layer call and reducing isolated median CPU time by 13.3%–27.7%; complete trace
  gains were a more modest 0.08%–6.23%.
- Composed the reference kernels into a narrow Float32 Llama-style full-prefill
  model whose layer states and logits match an independent deterministic oracle.
- Designed a deterministic versioned weight container with checked tensor geometry,
  aligned payloads, corruption checksums, and dual copied/memory-mapped loaders;
  enforced immutable mapped Tensor lifetimes and reduced warm-page median CPU load
  time by 12.78%–20.96% across three disclosed synthetic files.
- Built a bounded-memory Hugging Face Llama safetensors converter for sharded
  F32/F16/BF16 checkpoints, including strict architecture/path validation,
  projection transposition, RoPE coordinate remapping, atomic output replacement,
  and an independent bounded source-layout logits oracle; validated all 128,000
  logits from four raw-token sequences on one revision-pinned public checkpoint
  against both Helix and a CPU Float32 Transformers reference, then matched one
  eight-token cached greedy trace across Helix, full-prefix recomputation, and
  Transformers KV-cache decoding while rejecting broader compatibility claims.
- Built transactional single-sequence prefill/decode and greedy generation over a
  contiguous KV cache; matched cached logits/tokens against full-prefix
  recomputation and measured 3.69x–56.79x next-position CPU speedups across
  disclosed 8–64-token synthetic fixtures.

## Short project description

Built Helix to understand the low-level mechanisms beneath modern LLM inference
runtimes without delegating execution to PyTorch. The project emphasizes explicit
memory lifetime, correct asynchronous design, CPU/GPU oracle comparisons, honest
hardware-status labeling, and benchmark-controlled optimization.
