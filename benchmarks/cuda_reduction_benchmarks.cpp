#include "helix/kernels/cuda/reduction.hpp"
#include "helix/kernels/cuda/softmax.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"

#include <benchmark/benchmark.h>
#include <cstdint>

namespace helix::cuda::kernels {
namespace {

constexpr int kWarmupLaunches = 10;

void benchmark_reduction(benchmark::State& state, ReductionAlgorithm algorithm) {
  if (!runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  set_device(0);
  const auto rows = static_cast<Dimension>(state.range(0));
  const auto width = static_cast<Dimension>(state.range(1));
  const Tensor input =
      Tensor::random_uniform({rows, width}, 901, -1.0F, 1.0F, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::empty({rows}, DType::kFloat32, Device::cuda(0));
  Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    sum_last_dim_out(input, output, stream, false, algorithm);
  }
  stream.synchronize();

  Event start;
  Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    sum_last_dim_out(input, output, stream, false, algorithm);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }
  const std::int64_t elements = state.range(0) * state.range(1);
  state.SetItemsProcessed(state.iterations() * elements);
  state.SetBytesProcessed(state.iterations() * (elements + state.range(0)) *
                          static_cast<std::int64_t>(sizeof(float)));
}

void benchmark_softmax(benchmark::State& state, SoftmaxAlgorithm algorithm) {
  if (!runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  set_device(0);
  const auto rows = static_cast<Dimension>(state.range(0));
  const auto width = static_cast<Dimension>(state.range(1));
  const Tensor input =
      Tensor::random_uniform({rows, width}, 902, -8.0F, 8.0F, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::empty({rows, width}, DType::kFloat32, Device::cuda(0));
  Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    softmax_out(input, output, stream, algorithm);
  }
  stream.synchronize();

  Event start;
  Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    softmax_out(input, output, stream, algorithm);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }
  const std::int64_t elements = state.range(0) * state.range(1);
  state.SetItemsProcessed(state.iterations() * elements);
  state.SetBytesProcessed(state.iterations() * elements * 2 *
                          static_cast<std::int64_t>(sizeof(float)));
}

void BM_CudaReductionNaive(benchmark::State& state) {
  benchmark_reduction(state, ReductionAlgorithm::kNaive);
}

void BM_CudaReductionShared(benchmark::State& state) {
  benchmark_reduction(state, ReductionAlgorithm::kSharedMemory);
}

void BM_CudaReductionWarp(benchmark::State& state) {
  benchmark_reduction(state, ReductionAlgorithm::kWarpShuffle);
}

void BM_CudaSoftmaxNaive(benchmark::State& state) {
  benchmark_softmax(state, SoftmaxAlgorithm::kNaive);
}

void BM_CudaSoftmaxShared(benchmark::State& state) {
  benchmark_softmax(state, SoftmaxAlgorithm::kSharedMemory);
}

#define HELIX_REDUCTION_BENCHMARK(function)                                                        \
  BENCHMARK(function)->Args({32, 128})->Args({16, 1023})->Args({64, 4096})->UseManualTime()

HELIX_REDUCTION_BENCHMARK(BM_CudaReductionNaive);
HELIX_REDUCTION_BENCHMARK(BM_CudaReductionShared);
HELIX_REDUCTION_BENCHMARK(BM_CudaReductionWarp);
HELIX_REDUCTION_BENCHMARK(BM_CudaSoftmaxNaive);
HELIX_REDUCTION_BENCHMARK(BM_CudaSoftmaxShared);

#undef HELIX_REDUCTION_BENCHMARK

} // namespace
} // namespace helix::cuda::kernels
