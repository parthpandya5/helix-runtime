# Helix Performance Notebook

Helix publishes only measurements produced by executable benchmarks. This document
tracks the methodology and the conclusions from those measurements; it does not use
illustrative or estimated values in result columns.

## Required environment record

Every recorded experiment must include:

- date and Helix commit,
- CPU, RAM, and operating system,
- GPU and driver when applicable,
- CUDA toolkit and library versions when applicable,
- compiler and CMake versions,
- build type and relevant CMake options,
- dtype, tensor/model dimensions, batch size, and sequence length,
- warmup iterations, measured iterations, and repetitions,
- timing mechanism and synchronization boundary.

## Experiment log

| Experiment | Baseline | Change | Result | Interpretation |
|---|---|---|---|---|
| No optimization experiments recorded yet | — | — | — | Milestone 0 establishes the harness; performance work starts from executable subsystem baselines. |

## Timing rules

CPU microbenchmarks use Google Benchmark and a release build. Inputs are kept alive
with benchmark barriers where required, and setup/allocation is separated unless it
is the operation under study.

CUDA kernel timing uses warmed-up kernels and CUDA events recorded on the same
stream as the measured work. End-to-end host latency, transfer latency, and kernel
execution time are reported separately. A kernel launch measured without the
required synchronization is never reported as execution latency.

## Local benchmark commands

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DHELIX_ENABLE_CUDA=OFF
cmake --build build-release -j
./build-release/benchmarks/helix_benchmarks
```

The Milestone 0 dtype-metadata benchmark is a harness smoke test, not a runtime
performance claim.
