#include "helix/kernels/cuda/matmul.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"

#include <benchmark/benchmark.h>
#include <cstdint>

namespace helix::cuda::kernels {
namespace {

constexpr int kWarmupLaunches = 10;

void benchmark_matmul(benchmark::State& state, MatmulAlgorithm algorithm) {
  if (!runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  set_device(0);
  const auto rows = static_cast<Dimension>(state.range(0));
  const auto inner = static_cast<Dimension>(state.range(1));
  const auto columns = static_cast<Dimension>(state.range(2));
  const Tensor lhs =
      Tensor::random_uniform({rows, inner}, 1'101, -1.0F, 1.0F, DType::kFloat32, Device::cuda(0));
  const Tensor rhs = Tensor::random_uniform({inner, columns}, 1'102, -1.0F, 1.0F, DType::kFloat32,
                                            Device::cuda(0));
  Tensor output = Tensor::empty({rows, columns}, DType::kFloat32, Device::cuda(0));
  Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    matmul_out(lhs, rhs, output, stream, algorithm);
  }
  stream.synchronize();

  Event start;
  Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    matmul_out(lhs, rhs, output, stream, algorithm);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }

  const std::int64_t operations = 2 * state.range(0) * state.range(1) * state.range(2);
  state.SetItemsProcessed(state.iterations() * operations);
  state.counters["FLOP/s"] = benchmark::Counter(static_cast<double>(operations),
                                                benchmark::Counter::kIsIterationInvariantRate);
}

void BM_CudaMatmulNaive(benchmark::State& state) {
  benchmark_matmul(state, MatmulAlgorithm::kNaive);
}

void BM_CudaMatmulTiled(benchmark::State& state) {
  benchmark_matmul(state, MatmulAlgorithm::kTiled);
}

#define HELIX_MATMUL_BENCHMARK(function)                                                           \
  BENCHMARK(function)                                                                              \
      ->Args({31, 29, 37})                                                                         \
      ->Args({64, 64, 64})                                                                         \
      ->Args({128, 128, 128})                                                                      \
      ->Args({256, 256, 256})                                                                      \
      ->UseManualTime()

HELIX_MATMUL_BENCHMARK(BM_CudaMatmulNaive);
HELIX_MATMUL_BENCHMARK(BM_CudaMatmulTiled);

#undef HELIX_MATMUL_BENCHMARK

} // namespace
} // namespace helix::cuda::kernels
