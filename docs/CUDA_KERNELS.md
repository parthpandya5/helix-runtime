# CUDA Kernels

## Validation status

The Milestone 4 elementwise, Milestone 5 reduction/softmax, Milestone 6 SGEMM,
Milestone 9 transformer kernels, and Milestone 10 attention baseline are **CUDA
compile-validated, not GPU runtime-validated**. NVCC 12.6.85 and GCC 13.2 compiled
the implementations, hardware-gated
correctness tests, and CUDA-event benchmarks in the ARM64 CUDA 12.6 development
container with CUDA and host warnings promoted to errors. The container has no
NVIDIA driver or device, so the tests and benchmarks have not executed. This
document therefore contains no GPU latency, bandwidth, CPU/GPU crossover, or
speedup claim.

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
| SGEMM naive | 32 | 0 B | 0 B |
| SGEMM tiled 16x16 | 48 | 2,048 B | 0 B |
| RMSNorm warp-reduced | 21 | 32 B | 0 B |
| SwiGLU | 17 | 0 B | 0 B |
| RoPE adjacent-pair | 27 | 0 B | 0 B |
| attention score/scale/mask | 29 | 0 B | 0 B |
| attention probabilities x V | 32 | 0 B | 0 B |

This table helps make occupancy/resource hypotheses concrete, but occupancy cannot
be concluded from registers/shared memory alone; launch geometry, architecture
limits, instruction mix, and achieved scheduling must be measured on the target.

## SGEMM progression

Helix defines row-major `C[M,N] = A[M,K] * B[K,N]` for contiguous Float32 rank-two
tensors. Both custom paths compute one Float32 accumulator per output and reject
output aliasing with either input. The CPU oracle accumulates in double, so GPU
tests use an inner-dimension-aware tolerance. A fused multiply-add counts as two
floating-point operations; the benchmark reports `2*M*N*K` FLOPs.

### Version 1: one output per thread

A 16x16 block maps thread `(x,y)` to one output `(column,row)`. The two-dimensional
grid is capped independently in each dimension, and blocks/threads use grid-stride
loops so the mapping remains correct beyond the cap. Each thread walks all `K`
values and writes one output.

For a fixed reduction index, adjacent `x` lanes read adjacent values of `B` and
write adjacent values of `C`; their `A` value is common within a row and can use
the hardware's broadcast/cache behavior. However, the same `A` and `B` elements
are redundantly loaded by many output threads. Ignoring caches, one output performs
`2K` FLOPs, reads `2K` Float32 values, and writes one, for approximately
`2K / (8K + 4)` FLOP/byte. This is intentionally a transparent baseline.

### Version 2: 16x16 shared-memory tiling

Each block owns a 16x16 output tile. On every `K` tile:

1. threads cooperatively load one `A` tile and one `B` tile into two 16x16 shared
   arrays,
2. out-of-range edge positions contribute zero,
3. a block barrier makes both tiles visible,
4. each thread performs 16 multiply-adds from shared memory,
5. a second barrier prevents the next loads from replacing data still in use.

Both cooperative global loads are contiguous along `threadIdx.x`, including for
non-multiple dimensions before bounds masking. The compiler unrolls the fixed
16-step inner loop. A full tile performs `2*T^3` FLOPs from `2*T^2` Float32 global
loads, giving an ideal load-only arithmetic intensity of `T/4 = 4` FLOP/byte at
`T=16`, before accounting for the output write and cache effects. This model
explains the reuse opportunity; it is not a measured achieved bandwidth or speedup.

If `K` is zero and `M,N` are nonzero, both kernels still launch and write zero to
every output without dereferencing the null zero-byte input allocations. If `M` or
`N` is zero, there is no output storage and no launch. The tiled block's nested
grid-stride tile loops are uniform across all threads, so every barrier is reached
by the entire block.

### Correctness and measurement gate

The GPU suite covers a hand-computed 2x3 by 3x2 product; 1x1, 3x5x4, 17x19x23,
31x33x29, and zero-dimension problems; and invalid rank, inner dimension, dtype,
alias, and algorithm contracts. Both variants are compared elementwise with
`cpu::matmul`. These tests have compiled but not executed on a GPU.

The event benchmark performs ten warmups and times preallocated-output calls for
31x29x37, 64 cubed, 128 cubed, and 256 cubed shapes. A valid hardware study must
run the exact same problems, record GPU/driver/architecture, and compare the naive
and tiled results before describing one as faster. A cuBLAS SGEMM baseline will be
added and checked for equivalent row-major/precision semantics on that hardware;
the project does not imply that either educational kernel should beat cuBLAS.

The static `sm_52` compiler inspection reports 32 registers/thread for Version 1
and 48 registers/thread plus 2,048 bytes shared/block for Version 2. Those inputs
alone do not determine occupancy. Version 3 techniques such as register tiling,
vectorized loads, or alternative tile shapes are deliberately deferred until
profiling identifies a limiting resource.

## Transformer primitives

Milestone 9 adds Float32 RMSNorm, SwiGLU, and RoPE kernels in
`src/kernels/cuda/transformer.cu`. The bias-free linear API uses row-major weights
shaped `[input_features, output_features]` and delegates to the explicitly selected
M6 naive or tiled SGEMM. No cuBLAS or framework implementation is hidden behind
these entry points.

### RMSNorm

Input is interpreted as contiguous rows over the last dimension. One 256-thread
block owns a row and each thread accumulates squares at columns `threadIdx.x +
k*256`, giving adjacent lanes adjacent global loads. Eight warp-local shuffle sums
are reduced by the first warp through 32 bytes of shared memory. Thread zero stores
the reciprocal RMS in that shared array; after a barrier, all threads apply the
scale and per-column weight.

Exact input/output aliasing is safe because every thread finishes the collective
sum-of-squares read before any thread writes the row. Output/weight aliasing is
rejected. A final barrier prevents a grid-striding block from replacing shared
state while peers finish the previous row. Accumulation and `rsqrtf` are Float32;
the CPU oracle accumulates in double, so tests use a numerical tolerance rather
than bit equality. Empty outer dimensions suppress the launch, while a zero-width
row is rejected because its mean square is undefined.

This layout has modest static resource demand in the `sm_52` compiler inspection:
21 registers/thread, 32 bytes of shared memory, and no spills. That does not prove
occupancy or performance on a target GPU.

### SwiGLU

SwiGLU computes `silu(gate[i]) * up[i]` with a 256-thread grid-stride loop. It uses
the same overflow-resistant positive/negative SiLU branches as the standalone
elementwise implementation. Adjacent lanes read adjacent gate/up values and write
adjacent outputs, so the three arrays have coalescing-friendly access. Exact output
aliasing with either input is safe because a thread loads both values before its
single write. Static inspection reports 17 registers/thread and no shared memory or
spills.

### RoPE

Helix uses an explicit adjacent-pair convention over the complete even last
dimension:

```text
y[2p]     = x[2p] * cos[t,p] - x[2p+1] * sin[t,p]
y[2p + 1] = x[2p] * sin[t,p] + x[2p+1] * cos[t,p]
```

Input layout is `[tokens, ..., head_dimension]`; the intervening dimensions are
independent vectors/heads. Cosine and sine are preselected Float32 caches shaped
`[tokens, head_dimension/2]`. Selecting cache rows outside this kernel permits a
future packed batch to use unrelated token positions. It also makes the kernel a
pure rotation oracle without embedding an integer position representation into the
Tensor core.

One thread owns one pair. Pair index varies fastest, so adjacent threads consume
adjacent two-Float regions within a head; cache reads are adjacent and reused across
heads through normal caching. Each thread loads both coordinates before writing
either, making exact in-place rotation race-free. The grid-stride launch supports
odd pair-task counts and caps the grid at 65,535 blocks. Static `sm_52` inspection
reports 27 registers/thread, no shared memory, and no spills.

### Correctness and benchmark gate

CPU tests cover known quarter-turn rotations, three tokens by three heads with an
odd 27-pair task count, exact in-place execution, empty tokens, invalid rank, odd or
zero head width, cache shape, and output shape. GPU tests compare RMSNorm widths 1,
37, and 513; SwiGLU sizes 0, 1, 31, 257, and 4,099; RoPE shapes through 5x7x34;
both linear GEMM variants; exact in-place paths; and invalid contracts. The GPU
tests compile and are hardware-gated; they have not executed here.

CUDA benchmarks preallocate all storage, perform ten warmup launches, then use
same-stream CUDA events with stop-event synchronization. RMSNorm uses 1x4,096 and
32x4,096 rows, SwiGLU uses 4,096 and 131,072 values, and RoPE uses decode-shaped
1x32x128 and prefill-shaped 128x32x128 inputs. Projection uses the existing M6 GEMM
fixtures. No result or optimization conclusion is recorded without hardware.

## Contiguous attention baseline

Milestone 10 preserves score/scale/causal-mask, stable softmax, and
probability-times-V as three separate same-stream stages. The score and value
kernels are intentionally serial per output coordinate, while softmax reuses the
M5 shared-memory implementation. This exposes the `4*H*Q*K` Float32 workspace and
the naive score kernel's strided-across-warp K access rather than hiding either
behind a fused or library call. Static `sm_52` inspection reports 29 and 32
registers/thread for the custom stages with no shared memory or spills; the
softmax stage retains its documented 5,120-byte shared scratch.

CPU-oracle GPU tests cover known causal values/probabilities, one-token decode,
odd dimensions through head width 37/value width 41, empty queries, contracts, and
alias rejection. Ten-warmup event fixtures time the full three-stage sequence at
decode, odd, and prefill shapes, but have not executed. Tensor layout, causal
offset semantics, memory traffic, CPU baselines, and limitations are detailed in
`docs/ATTENTION.md`.

## Reproducible profiling

`scripts/profile_cuda.sh` turns one exact CUDA Google Benchmark case into a
bounded Nsight Systems timeline, Nsight Compute kernel report, or both. It records
the selected benchmark, exact command, launch bounds, source revision, build
settings, CPU/RAM, CUDA/driver/profiler versions, and GPU inventory beside the
capture. `HELIX_ENABLE_CUDA_LINEINFO=ON` emits CUDA line information for source
correlation. The runner performs no collection in `--dry-run`, rejects unsafe
report stems and invalid counts, requires exactly one registered CUDA case,
refuses accidental overwrite, and will not collect until both `nvidia-smi` and a
one-iteration benchmark preflight demonstrate CUDA visibility.

All single-kernel fixtures currently perform ten explicit warmup launches. The
default Nsight Compute skip/count pair is therefore 10/1. Contiguous attention
launches score/mask, shared softmax, and probability-times-V for each operation;
profiling the first complete measured operation requires skip/count 30/3. Nsight
Compute replay output is not accepted as a latency benchmark. See the complete
commands and result-publication gate in
[`docs/PERFORMANCE.md`](PERFORMANCE.md#cuda-profiling-workflow).

## Planned evolution

The caching device allocator retains direct `cudaMalloc`/`cudaFree` as its benchmark
baseline. A later production projection path may select clearly named cuBLAS while
keeping both custom GEMMs for education and measurement. No variant will be called
faster without comparable event-timed measurements.
