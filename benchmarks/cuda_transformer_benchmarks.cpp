#include "helix/kernels/cpu/reference.hpp"
#include "helix/kernels/cuda/transformer.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace helix::cuda::kernels {
namespace {

constexpr int kWarmupLaunches = 10;

bool require_runtime(benchmark::State& state) {
  if (runtime_available()) {
    return true;
  }
  state.SkipWithError("requires a visible NVIDIA CUDA device");
  return false;
}

void record_elements(benchmark::State& state, std::int64_t elements, std::int64_t logical_tensors) {
  state.SetItemsProcessed(state.iterations() * elements);
  state.SetBytesProcessed(state.iterations() * elements * logical_tensors *
                          static_cast<std::int64_t>(sizeof(float)));
}

void BM_CudaRmsNorm(benchmark::State& state) {
  if (!require_runtime(state)) {
    return;
  }
  set_device(0);
  const auto rows = static_cast<Dimension>(state.range(0));
  const auto width = static_cast<Dimension>(state.range(1));
  const Tensor input =
      Tensor::random_uniform({rows, width}, 1'201, -2.0F, 2.0F, DType::kFloat32, Device::cuda(0));
  const Tensor weight =
      Tensor::random_uniform({width}, 1'202, 0.5F, 1.5F, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::empty({rows, width}, DType::kFloat32, Device::cuda(0));
  Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    rms_norm_out(input, weight, 1.0e-5F, output, stream);
  }
  stream.synchronize();

  Event start;
  Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    rms_norm_out(input, weight, 1.0e-5F, output, stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }
  record_elements(state, state.range(0) * state.range(1), 3);
}

void BM_CudaSwiGlu(benchmark::State& state) {
  if (!require_runtime(state)) {
    return;
  }
  set_device(0);
  const auto elements = static_cast<Dimension>(state.range(0));
  const Tensor gate =
      Tensor::random_uniform({elements}, 1'301, -8.0F, 8.0F, DType::kFloat32, Device::cuda(0));
  const Tensor up =
      Tensor::random_uniform({elements}, 1'302, -2.0F, 2.0F, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::empty({elements}, DType::kFloat32, Device::cuda(0));
  Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    swiglu_out(gate, up, output, stream);
  }
  stream.synchronize();

  Event start;
  Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    swiglu_out(gate, up, output, stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }
  record_elements(state, state.range(0), 3);
}

void BM_CudaRope(benchmark::State& state) {
  if (!require_runtime(state)) {
    return;
  }
  set_device(0);
  const auto tokens = static_cast<Dimension>(state.range(0));
  const auto heads = static_cast<Dimension>(state.range(1));
  const auto head_dimension = static_cast<Dimension>(state.range(2));
  const Tensor input = Tensor::random_uniform({tokens, heads, head_dimension}, 1'401, -2.0F, 2.0F,
                                              DType::kFloat32, Device::cuda(0));
  const auto cache_elements =
      static_cast<std::size_t>(tokens) * static_cast<std::size_t>(head_dimension / 2);
  std::vector<float> cosine_values(cache_elements);
  std::vector<float> sine_values(cache_elements);
  for (std::size_t index = 0; index < cache_elements; ++index) {
    const float angle = static_cast<float>(index) * 0.001F;
    cosine_values[index] = std::cos(angle);
    sine_values[index] = std::sin(angle);
  }
  const Tensor cosine =
      Tensor::from_values({tokens, head_dimension / 2}, cosine_values, Device::cuda(0));
  const Tensor sine =
      Tensor::from_values({tokens, head_dimension / 2}, sine_values, Device::cuda(0));
  Tensor output = Tensor::empty(input.shape(), DType::kFloat32, Device::cuda(0));
  Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    rope_out(input, cosine, sine, output, stream);
  }
  stream.synchronize();

  Event start;
  Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    rope_out(input, cosine, sine, output, stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }
  record_elements(state, state.range(0) * state.range(1) * state.range(2), 3);
}

template <bool TokenToHead> void run_cuda_head_layout(benchmark::State& state) {
  if (!require_runtime(state)) {
    return;
  }
  set_device(0);
  const auto tokens = static_cast<Dimension>(state.range(0));
  const auto heads = static_cast<Dimension>(state.range(1));
  const auto head_dimension = static_cast<Dimension>(state.range(2));
  const Tensor token_input =
      Tensor::random_uniform({tokens, heads, head_dimension}, 1'501, -2.0F, 2.0F);
  const Tensor head_input = helix::cpu::token_heads_to_head_major(token_input);
  const Tensor input_host = TokenToHead ? token_input : head_input;
  const Tensor expected = TokenToHead ? head_input : token_input;
  const Tensor input = input_host.to(Device::cuda(0));
  Tensor output = Tensor::empty(expected.shape(), DType::kFloat32, Device::cuda(0));
  Stream stream;
  const auto execute = [&input, &output, &stream]() {
    if constexpr (TokenToHead) {
      token_heads_to_head_major_out(input, output, stream);
    } else {
      head_major_to_token_heads_out(input, output, stream);
    }
  };
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    execute();
  }
  stream.synchronize();
  const auto output_matches_or_skip = [&state, &output, &expected]() {
    const Tensor actual = output.to(Device::cpu());
    for (std::size_t index = 0; index < expected.numel(); ++index) {
      if (actual.data_as<float>()[index] != expected.data_as<float>()[index]) {
        state.SkipWithError("CUDA head-layout output does not match the CPU oracle");
        return false;
      }
    }
    return true;
  };
  if (!output_matches_or_skip()) {
    return;
  }

  Event start;
  Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    execute();
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }
  if (!output_matches_or_skip()) {
    return;
  }
  record_elements(state, state.range(0) * state.range(1) * state.range(2), 2);
  state.counters["tokens"] = static_cast<double>(state.range(0));
  state.counters["heads"] = static_cast<double>(state.range(1));
  state.counters["head_dimension"] = static_cast<double>(state.range(2));
}

void BM_CudaTokenHeadsToHeadMajor(benchmark::State& state) {
  run_cuda_head_layout<true>(state);
}

void BM_CudaHeadMajorToTokenHeads(benchmark::State& state) {
  run_cuda_head_layout<false>(state);
}

BENCHMARK(BM_CudaRmsNorm)->Args({1, 4096})->Args({32, 4096})->UseManualTime();
BENCHMARK(BM_CudaSwiGlu)->Arg(4096)->Arg(32 * 4096)->UseManualTime();
BENCHMARK(BM_CudaRope)->Args({1, 32, 128})->Args({128, 32, 128})->UseManualTime();
BENCHMARK(BM_CudaTokenHeadsToHeadMajor)
    ->Args({1, 32, 128})
    ->Args({128, 32, 128})
    ->Args({7, 3, 37})
    ->Args({257, 13, 65})
    ->UseManualTime();
BENCHMARK(BM_CudaHeadMajorToTokenHeads)
    ->Args({1, 32, 128})
    ->Args({128, 32, 128})
    ->Args({7, 3, 37})
    ->Args({257, 13, 65})
    ->UseManualTime();

} // namespace
} // namespace helix::cuda::kernels
