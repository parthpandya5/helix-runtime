# CUDA Kernels

## Validation status

The Milestone 4 elementwise kernels are **CUDA compile-validated, not GPU
runtime-validated**. NVCC 12.6.85 and GCC 13.2 compiled the implementation,
hardware-gated correctness tests, and CUDA-event benchmarks in the ARM64 CUDA
12.6 development container with warnings as errors. The container has no NVIDIA
driver or device, so the tests and benchmarks have not executed. This document
therefore contains no GPU latency, bandwidth, CPU/GPU crossover, or speedup claim.

## Implemented operations

Helix manually implements contiguous Float32 add, multiply, ReLU, and SiLU in
`src/kernels/cuda/elementwise.cu`. These kernels do not call cuBLAS, cuDNN, or a
tensor framework. Their allocating APIs create an output Tensor; their `_out`
forms accept preallocated storage and are the benchmark path.

Each host entry point validates before launch:

- all tensors are CUDA-resident, contiguous, and Float32,
- binary input shapes and devices match,
- output shape and device match the input,
- the stream belongs to the tensor's device,
- an empty tensor returns without an invalid zero-block launch.

Input/output aliasing is safe for these pointwise definitions because each thread
loads and stores only the same logical element. Partial overlapping views cannot
currently be constructed by the public contiguous Tensor API and are not part of
the contract.

## Thread and memory layout

```mermaid
flowchart LR
    T[CUDA thread] --> I[initial linear index]
    I --> L{index < numel?}
    L -->|yes| R[read contiguous input element(s)]
    R --> W[write matching output element]
    W --> S[index += grid size]
    S --> L
```

Blocks contain 256 threads. The grid contains enough blocks for the input, capped
at 65,535, and every thread advances by `gridDim.x * blockDim.x`. The cap keeps the
launch one-dimensional and portable while the grid-stride loop supports tensors
larger than one grid. Size arithmetic is performed in `std::size_t`; ceiling
division uses `1 + (n - 1) / block_size` only after proving `n != 0`, avoiding the
usual `n + block_size - 1` overflow.

Within a warp, adjacent threads access adjacent Float32 elements. The first
iteration therefore produces naturally coalesced global-memory accesses, and later
grid-stride iterations preserve that adjacency. Add and multiply read two arrays
and write one; ReLU and SiLU read one and write one. They use no shared memory and
require no inter-thread synchronization. Their useful arithmetic per transferred
byte is low, so sufficiently large inputs are expected to be memory-bandwidth
limited; this remains a hypothesis until hardware measurement.

Register demand is intentionally small: a linear index, grid stride, and operation
temporaries. The fixed 256-thread block is a conventional baseline, not an
empirically selected optimum. Launch bounds, vectorized loads, and fused operations
are deferred until profiling identifies a bottleneck.

## Numerical behavior

Add, multiply, and ReLU follow the Float32 CPU oracle element by element. ReLU maps
negative values to zero. SiLU avoids overflow in its sigmoid calculation:

- for nonnegative `x`, it evaluates `x / (1 + exp(-x))`;
- for negative `x`, it evaluates `x * exp(x) / (1 + exp(x))`.

This is mathematically equivalent to `x * sigmoid(x)` for finite inputs but avoids
forming `exp(-x)` for large negative values. GPU tests include values from -100 to
100 and use a tolerance appropriate for CPU/GPU transcendental implementation
differences.

## Asynchrony and lifetime

Kernel launches are asynchronous on the caller-owned nonblocking stream. The
entry points establish the stream's CUDA device with `DeviceGuard` and call
`cudaGetLastError` immediately after launch to surface configuration and launch
errors with the kernel name. They do not synchronize. The caller must retain all
input and output storage until stream completion or a recorded event proves the
work finished. Current direct Tensor storage uses `cudaFree`, but a future caching
allocator must preserve this stream-ordered lifetime explicitly.

## Correctness tests

`tests/cuda_elementwise_test.cpp` constructs deterministic CPU inputs, computes the
CPU reference, transfers inputs, launches the CUDA operation, synchronizes, copies
the result back, and compares every element. Parameterized sizes are 0, 1, 31,
257, 4,099, and 65,539, covering empty work, warp/block boundaries, and non-block-
aligned grids. Separate tests cover exact in-place add, extreme finite SiLU values,
shape mismatch, and unsupported dtype rejection.

The tests are labeled `gpu`; compile-only CI builds and lists them but runs only
hardware-independent tests. On a CUDA host:

```bash
ctest --test-dir build-cuda -L gpu --output-on-failure
compute-sanitizer --tool memcheck ./build-cuda/tests/helix_cuda_elementwise_tests
```

## Benchmark methodology

`BM_CudaAdd` and `BM_CudaSilu` preallocate inputs/output, perform ten warmup
launches, and measure each kernel with timing-enabled CUDA events recorded on the
same stream. The stop event is synchronized before reading elapsed time, so launch
enqueue time is not mistaken for execution time. Allocation and host/device copies
are outside the timed region. Bytes processed are reported from explicit tensor
traffic (three Float32 values per add element and two per SiLU element).

CPU and CUDA benchmarks share element counts 1, 1,024, and 1,048,576. The
single-element case is specifically present to expose launch overhead. Results and
crossover interpretation will be added to `docs/PERFORMANCE.md` only after both
paths execute on identified hardware under a reproducible Release build.

## Planned evolution

Milestone 5 adds independently testable reduction and stable row-wise softmax
baselines before exploring shared-memory and warp-assisted variants. Milestone 6
adds naive and shared-memory tiled GEMM as distinct selectable implementations.
Optimized variants will not replace their educational baselines, and no variant
will be called faster without comparable event-timed measurements.
