# CUDA Kernels

## Validation status

The Milestone 4 elementwise and Milestone 5 reduction/softmax kernels are **CUDA
compile-validated, not GPU runtime-validated**. NVCC 12.6.85 and GCC 13.2 compiled
the implementations, hardware-gated correctness tests, and CUDA-event benchmarks
in the ARM64 CUDA 12.6 development container with CUDA and host warnings promoted
to errors. The container has no NVIDIA driver or device, so the tests and
benchmarks have not executed. This document therefore contains no GPU latency,
bandwidth, CPU/GPU crossover, or speedup claim.

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

## Elementwise correctness tests

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

## Elementwise benchmark methodology

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

## Last-dimension reduction

`sum_last_dim` treats every contiguous slice along the final dimension as one row
and offers three independently selectable algorithms. All use Float32 device
accumulation; the CPU oracle uses a double accumulator, so correctness tests use a
width-aware tolerance rather than requiring bit identity.

### Naive

One CUDA thread owns one row and sums its columns serially. Threads use a grid-
stride loop over rows. The implementation needs no shared memory or block-wide
synchronization, but adjacent threads walk different rows, so loads within a warp
are separated by the row width. It is the clearest correctness baseline and is
expected to underuse parallelism for wide rows; that expectation is not a measured
result.

### Shared-memory tree

One 256-thread block owns a row. Thread `t` accumulates columns `t`, `t + 256`, and
so on, which makes each warp's loads contiguous and coalescing-friendly. Threads
write partial sums into 1,024 bytes of dynamic shared memory and perform a binary
tree reduction with a barrier at each stage. A final barrier protects the shared
array before a block advances to another row through its grid-stride loop.

### Warp-shuffle tree

The third variant performs a shuffle reduction within each of eight warps, stores
one Float32 sum per warp in 32 bytes of shared memory, and has the first warp reduce
those eight values. It removes most shared-memory traffic and block barriers from
the tree while preserving the same row/thread mapping. This is an optimization
hypothesis retained alongside both baselines; only event-timed hardware data can
establish whether it helps for a particular row width and GPU.

Widths may be zero for sum: every algorithm then writes the additive identity.
Zero output rows return without launching. Reductions reject output/input aliasing,
including the otherwise shape-compatible width-one `keep_dim` case, because that
contract will remain safe when implementations evolve asynchronously.

## Stable row-wise softmax

Softmax operates over the last dimension and requires a nonzero width. The naive
variant assigns one thread per row and makes serial passes for the maximum,
denominator, and normalization. The block variant assigns one 256-thread block per
row. Its threads cooperatively reduce the maximum and non-finite counters, reduce
the exponential sum, then normalize strided columns. Adjacent lanes access
adjacent values on every cooperative pass.

The block implementation uses 5,120 bytes of static shared memory: 256 Float32
scratch values and two 256-entry `std::size_t` counter arrays. The wide counters
avoid silently overflowing the count of positive infinities on exceptionally wide
rows. Both implementations subtract the row maximum before exponentiation and
match the CPU oracle's explicit non-finite semantics:

- a row containing NaN produces an all-NaN row,
- an all-negative-infinity row produces an all-NaN row,
- positive infinity mass is divided uniformly among positive-infinity positions.

Exact input/output aliasing is safe. Every input needed for collective statistics
is read before any finite-row output write, and each thread later reads and writes
only its assigned positions. Partial overlap is outside the current Tensor API.

## Reduction and softmax correctness tests

Every reduction variant is compared with `cpu::sum_last_dim` at widths 0, 1, 31,
257, and 1,023; the suite also covers `keep_dim`, multiple ranks, alias rejection,
dtype/rank errors, and invalid algorithms even when no kernel would launch. Both
softmax variants are compared with `cpu::softmax` at zero-row and odd-width shapes,
exercise width 513 with values in [-80, 80], verify row normalization, test exact
in-place execution, and check NaN/infinity behavior. These tests compile and are
labeled `gpu`; they have not run in the current environment.

## Reduction and softmax benchmark methodology

All five variant/operation pairs use preallocated outputs, ten warmup launches, and
same-stream CUDA event timing with explicit stop-event synchronization. Shapes are
32x128, 16x1,023, and 64x4,096. The benchmark reports logical values and minimum
explicit input/output bytes; it does not pretend this is the exact internal memory
traffic of multi-pass softmax. Results remain pending GPU hardware.

## Static compiler resource inspection

`cuobjdump --dump-resource-usage` was run on the CUDA 12.6 Release archive compiled
for the container's default `sm_52` target. These are compiler outputs, not runtime
measurements, and may change with architecture/toolchain:

| Kernel | Registers/thread | Static shared | Dynamic shared at launch |
|---|---:|---:|---:|
| reduction naive | 20 | 0 B | 0 B |
| reduction shared tree | 17 | 0 B | 1,024 B |
| reduction warp shuffle | 18 | 32 B | 0 B |
| softmax naive | 33 | 0 B | 0 B |
| softmax shared | 25 | 5,120 B | 0 B |

This table helps make occupancy/resource hypotheses concrete, but occupancy cannot
be concluded from registers/shared memory alone; launch geometry, architecture
limits, instruction mix, and achieved scheduling must be measured on the target.

## Planned evolution

Milestone 6 adds naive and shared-memory tiled GEMM as distinct selectable
implementations. Optimized variants will not replace their educational baselines,
and no variant will be called faster without comparable event-timed measurements.
