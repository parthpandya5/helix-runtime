# Helix

**A C++/CUDA LLM inference runtime built from first principles.**

Helix explores the systems underneath modern decoder-only LLM inference: explicit
tensor/storage ownership, CPU correctness kernels, manually implemented CUDA
kernels, GPU allocation and asynchronous lifetime, paged KV cache, attention, and
continuous request batching.

The project is developed in correctness and measurement gates. Features are listed
as complete only after their applicable tests pass; optimization claims require
reproducible benchmark evidence.

## Current status

Milestones 0–2, the host-testable portions of Milestones 7–8, and the CPU path of
Milestone 9 are implemented and CPU-validated. The repository contains the
CPU-first build foundation, a checked contiguous tensor core, Float32 reference
kernels for elementwise math, reductions, stable softmax, matrix multiplication,
RMSNorm, SwiGLU, adjacent-pair RoPE, and linear projection, plus a caching allocator
and completion-aware deferred reclamation. On 2026-08-20, Apple Clang 17 built the
project on macOS arm64 with warnings as errors; 72/72 CPU tests passed in normal and
ASan/UBSan configurations.

CUDA is optional at configure time. The current Apple Silicon development host has
no NVIDIA GPU or native CUDA toolkit, so no CUDA execution or performance claim is
made. The M3 runtime layer, M4 manual Float32 elementwise kernels, M5 selectable
reduction/stable-softmax baselines, and M6 naive/tiled SGEMM are separately
compile-validated in an ARM64 CUDA 12.6 development container without a
driver/device. The M8 stream-ordered allocator and M9 block-reduced RMSNorm,
grid-stride SwiGLU, adjacent-pair RoPE, and projection adapter are also compile-
validated, not runtime-validated.

## Architecture

```mermaid
flowchart TD
    Request --> Scheduler --> Batch --> Executor[Model Executor]
    Executor --> Tensor[Tensor Runtime]
    Executor --> Allocator[HelixAllocator]
    Executor --> Kernels[CPU / CUDA Kernels]
    Executor --> KV[Paged KV Cache]
    Allocator --> Memory[(GPU Memory)]
    Kernels --> Memory
    KV --> Memory
```

The full dependency, ownership, request-lifecycle, and memory designs are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Build

Dependencies are discovered from the system first. When GoogleTest or Google
Benchmark is missing, CMake fetches the pinned source release by default. Set
`HELIX_FETCH_DEPENDENCIES=OFF` for fully offline/system-package builds.

### CPU-only

```bash
cmake -S . -B build -DHELIX_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Or run the complete local CPU validation:

```bash
./scripts/run_cpu_validation.sh
```

### CUDA

On a machine with a CUDA compiler and toolkit:

```bash
cmake -S . -B build-cuda -DHELIX_ENABLE_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

The CUDA-on build contains checked device, stream, event, allocation, copy, and
Tensor-storage paths, manual elementwise kernels, CUDA-event benchmarks, and
hardware-gated correctness tests. Reduction includes naive, shared-memory, and
warp-shuffle paths; softmax includes naive and shared-memory paths; SGEMM includes
one-output-per-thread and 16x16 tiled paths. Transformer primitives add custom
RMSNorm, SwiGLU, and RoPE kernels while projection delegates to the selectable
custom SGEMM. The build is compile-validated, but those tests and benchmarks have
not executed on a physical NVIDIA GPU.

### Sanitizers and analysis

```bash
cmake -S . -B build-sanitize \
  -DHELIX_ENABLE_CUDA=OFF \
  -DHELIX_ENABLE_SANITIZERS=ON
cmake --build build-sanitize -j
ctest --test-dir build-sanitize --output-on-failure

cmake --build build --target format-check
```

`HELIX_ENABLE_CLANG_TIDY=ON` requires `clang-tidy` in `PATH` and applies it only to
Helix targets.

## Benchmarks

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DHELIX_ENABLE_CUDA=OFF
cmake --build build-release -j
./build-release/benchmarks/helix_benchmarks
```

The first release-mode `Tensor::zeros` baseline is recorded without an optimization
claim. Allocation-free CPU reference-kernel baselines are also recorded, including
odd-width softmax, odd-dimension/standard-square matmul, SwiGLU, and decode/prefill-
shaped RoPE. Methodology and actual measurements live in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Tensor runtime

`helix::Tensor` is a shallow-copy handle over shared RAII storage. It tracks a
signed 64-bit shape and canonical row-major strides, dtype, device, logical bytes,
storage bytes, and byte offset. Size multiplication is checked before allocation.
`reshape`/`view` share storage only when safe; `clone` and same-device `to` create
independent storage. Float32 is executable, while Float16 currently supports only
metadata, allocation, zeroing, and bytewise copy.

```bash
cmake --build build --target helix_tensor_demo
./build/examples/helix_tensor_demo
```

## CPU reference kernels

The `helix::cpu` backend provides add, multiply, ReLU, SiLU, full and last-axis
sum, row-wise softmax, rank-two matmul, RMSNorm, SwiGLU, adjacent-pair RoPE, and a
model-layout linear wrapper. Each operation has a convenient allocating form and a
preallocated `_out` form used for benchmarking and GPU-oracle comparisons.
Broadcasting is deliberately rejected.

Reductions, GEMM, softmax denominators, and RMS statistics use wider accumulation
where helpful. Softmax subtracts the row maximum and explicitly defines NaN,
positive-infinity, and all-negative-infinity behavior.

```bash
cmake --build build --target helix_cpu_kernels_demo
./build/examples/helix_cpu_kernels_demo
```

## Caching allocator

`helix::memory::HelixAllocator` is a backend-neutral, mutex-protected best-fit
segment cache with alignment rounding, exact block splitting, adjacent coalescing,
allocator-instance and generation-tagged handles, reservation limits, explicit
trimming, and internal/external fragmentation statistics. `DeferredAllocator`
keeps logically freed allocations active until a completion object reports ready;
`StreamOrderedAllocator` implements that completion with timing-disabled CUDA
events recorded on an explicit release stream. The allocator/deferred metadata
passes 18 focused CPU and sanitizer tests. CUDA adapters and native-CUDA comparison
fixtures compile, but have not run on a GPU.

The allocator is not yet used by Tensor CUDA storage. Tensor does not yet record
every stream that uses its storage, so it cannot prove that a chosen release event
orders after all outstanding work. Design, measured host baselines, stream-ordering
contract, and limitations are in
[`docs/MEMORY_ALLOCATOR.md`](docs/MEMORY_ALLOCATOR.md).

## Supported hardware

- CPU foundation: validated with Apple Clang 17 on macOS arm64; Linux GCC/Clang
  validation is configured in CI.
- CUDA backend: requires a compatible NVIDIA CUDA toolkit; runtime hardware
  validation is pending and excluded from CPU CI.
- NCCL/multi-GPU: not implemented.

## Limitations

The CUDA runtime, M4–M6/M9 kernels, and M7–M8 CUDA allocator adapters are
compile-validated only; physical GPU correctness, timing, and resource validation
remain pending. Tensor allocator integration, attention, KV cache, scheduler,
model execution, and generation are roadmap work and are not represented as
complete. Float16 has no arithmetic path yet. Tensor execution is contiguous-only;
broadcasting, slicing, and general strided kernels are intentionally deferred. CPU
kernels are reference implementations, not optimized BLAS replacements.

## Engineering notes

- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap and validation gates](docs/ROADMAP.md)
- [CUDA runtime ownership and validation](docs/CUDA_RUNTIME.md)
- [CUDA kernel design and validation](docs/CUDA_KERNELS.md)
- [Caching allocator design and fragmentation](docs/MEMORY_ALLOCATOR.md)
- [Performance methodology](docs/PERFORMANCE.md)
- [Contributor/agent rules](AGENTS.md)

Helix is independently implemented. Design inspiration may come from CUDA
documentation and public inference-system literature, but external source code is
not copied into the runtime.
