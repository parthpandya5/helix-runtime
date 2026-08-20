# Helix Engineering Guide

This file defines the working rules for every contributor and coding agent in this
repository. Read it together with `docs/ARCHITECTURE.md` and `docs/ROADMAP.md`
before changing a subsystem.

## Mission

Helix is a C++20/CUDA inference runtime built to make the mechanisms beneath
decoder-only LLM inference understandable. The repository favors a smaller set of
correct, tested, benchmarked systems over a large set of shallow features.

Helix is not a training framework, a PyTorch wrapper, an HTTP service, or a clone
of another runtime. Implementations must be original. Cite conceptual inspiration
when it materially shapes a design.

## Non-negotiable rules

1. Preserve the CPU-only build. A machine without a CUDA toolkit must be able to
   configure, build, and run the CPU test suite.
2. Work in milestone order. Do not use a later feature to hide a missing or broken
   earlier layer.
3. For each performance-sensitive subsystem, use this sequence:
   correctness tests, baseline, benchmark, optimization, benchmark again,
   documentation.
4. Never invent benchmark results. Record the hardware, toolchain, build type,
   workload, warmups, repetitions, and precision with every published number.
5. Never describe compile-only CUDA validation as GPU runtime validation.
6. Keep custom/educational implementations distinct from NVIDIA-library-backed
   production paths in APIs, benchmarks, and documentation.
7. Do not add empty subsystem directories or placeholder features merely to make
   the project appear broader.
8. No autograd, training stack, web dashboard, authentication, Kubernetes, or
   mandatory Python runtime belongs in the core project.

## Required development loop

Before implementing a subsystem:

- Read its architecture and roadmap sections.
- Define ownership, invariants, failure behavior, and thread-safety.
- Add deterministic tests, including boundary and odd-sized cases.
- Implement the simplest correct CPU/reference path where applicable.
- Build and run the relevant tests before proceeding.

Before optimizing:

- Establish a reproducible baseline.
- State a bottleneck hypothesis.
- Change one important variable at a time when practical.
- Measure again under comparable conditions.
- Keep useful negative results in `docs/PERFORMANCE.md`.

## Build and validation

The canonical CPU commands are:

```bash
cmake -S . -B build -DHELIX_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The canonical CUDA commands, on a machine with the CUDA toolkit, are:

```bash
cmake -S . -B build-cuda -DHELIX_ENABLE_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

Useful checks are:

```bash
cmake --build build --target format-check
cmake -S . -B build-sanitize \
  -DHELIX_ENABLE_CUDA=OFF \
  -DHELIX_ENABLE_SANITIZERS=ON
cmake --build build-sanitize -j
ctest --test-dir build-sanitize --output-on-failure
```

Run the narrowest relevant tests during iteration and the full applicable suite
before declaring a milestone complete.

## Code organization

- Public headers live under `include/helix/<subsystem>/`.
- Implementations live under `src/<subsystem>/`.
- CUDA translation units use `.cu` and are only added when
  `HELIX_ENABLE_CUDA=ON`.
- Tests mirror subsystem names under `tests/`.
- Benchmarks live under `benchmarks/` and must identify their workload clearly.
- Examples are added only after the demonstrated subsystem is usable.
- Dependency direction follows `docs/ARCHITECTURE.md`; avoid circular coupling.

## C++ conventions

- Use C++20, RAII, explicit ownership, move semantics, `const` correctness, and
  checked integer arithmetic for sizes.
- Prefer value types and composition over class hierarchies.
- Do not use raw `new` or `delete` in application code.
- A non-owning pointer or view must be visibly non-owning in its type or API.
- Do not silently narrow sizes, dimensions, offsets, or device indices.
- Error messages must include the operation and relevant context.
- Public APIs belong in the `helix` namespace; implementation-only helpers belong
  in `helix::detail` or an unnamed namespace.
- Use `snake_case` for functions and variables, `PascalCase` for types, and
  `kPascalCase` for constants.
- Follow `.clang-format`; keep `.clang-tidy` suppressions local and justified.

## Tensor and kernel invariants

- Tensor shapes and strides use signed 64-bit dimensions so invalid negative
  inputs can be rejected before conversion to byte counts.
- Element-count and byte-count multiplication must detect overflow.
- A contiguous tensor has canonical row-major strides.
- Views share storage ownership and may not outlive their backing allocation.
- Kernel APIs validate device, dtype, shape, contiguity, and aliasing assumptions
  at the host boundary.
- Every CUDA mathematical kernel is compared with a deterministic CPU oracle using
  dtype-appropriate tolerances.
- CUDA tests include tiny, zero-work where supported, odd, and non-block-aligned
  dimensions.

## CUDA rules

- Check every CUDA runtime/library call and preserve the failed expression, file,
  line, device, byte count, or stream context where useful.
- CUDA handles, streams, events, and allocations use RAII.
- Kernel launches are followed by launch-error checks; synchronization is added
  only where required for correctness or measurement.
- GPU timing uses CUDA events around warmed-up work. Host wall-clock launch timing
  is not reported as kernel execution time.
- Memory freed on one stream cannot be reused while another stream may still
  access it. Stream-aware lifetime work must include a test that exercises this
  invariant on real GPU hardware.

## Thread safety

Thread safety is opt-in and documented per type. Immutable metadata and independent
tensor objects may be used concurrently. Mutable allocator, scheduler, request,
and cache state is not presumed thread-safe. Add synchronization only around an
identified shared-state boundary and test it.

## Documentation status language

Use these labels consistently:

- **Implemented**: code exists and applicable local correctness tests pass.
- **Compile-validated**: the path compiled but did not run on target hardware.
- **Hardware-validated**: correctness tests ran on the stated hardware.
- **Benchmarked**: a reproducible measurement is recorded with its environment.
- **Planned**: no completion claim is implied.

Update `README.md`, `docs/ROADMAP.md`, and the subsystem document whenever a
feature's status changes. Resume bullets must describe only implemented work;
quantitative claims require recorded benchmark evidence.

## Commit hygiene

Keep commits scoped to one coherent change and use descriptive prefixes such as
`build:`, `feat(tensor):`, `test(kernels):`, `bench(memory):`, or `docs:`. Do not
rewrite or discard unrelated user changes. Do not commit generated build trees,
downloaded model weights, profiler captures, or benchmark binaries.
