#include "helix/kernels/cuda/elementwise.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"

#include <benchmark/benchmark.h>
#include <cstdint>

namespace helix::cuda::kernels {
namespace {

constexpr int kWarmupLaunches = 10;

void BM_CudaAdd(benchmark::State& state) {
  if (!runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  set_device(0);
  const auto elements = static_cast<Dimension>(state.range(0));
  const Tensor lhs =
      Tensor::random_uniform({elements}, 101, -1.0F, 1.0F, DType::kFloat32, Device::cuda(0));
  const Tensor rhs =
      Tensor::random_uniform({elements}, 202, -1.0F, 1.0F, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::empty({elements}, DType::kFloat32, Device::cuda(0));
  Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    add_out(lhs, rhs, output, stream);
  }
  stream.synchronize();

  Event start;
  Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    add_out(lhs, rhs, output, stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) * 3 *
                          static_cast<std::int64_t>(sizeof(float)));
}

void BM_CudaSilu(benchmark::State& state) {
  if (!runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  set_device(0);
  const auto elements = static_cast<Dimension>(state.range(0));
  const Tensor input =
      Tensor::random_uniform({elements}, 303, -8.0F, 8.0F, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::empty({elements}, DType::kFloat32, Device::cuda(0));
  Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    silu_out(input, output, stream);
  }
  stream.synchronize();

  Event start;
  Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    silu_out(input, output, stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) * 2 *
                          static_cast<std::int64_t>(sizeof(float)));
}

BENCHMARK(BM_CudaAdd)->Arg(1)->Arg(1024)->Arg(1 << 20)->UseManualTime();
BENCHMARK(BM_CudaSilu)->Arg(1)->Arg(1024)->Arg(1 << 20)->UseManualTime();

} // namespace
} // namespace helix::cuda::kernels
