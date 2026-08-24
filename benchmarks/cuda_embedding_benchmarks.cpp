#include "helix/kernels/cpu/reference.hpp"
#include "helix/kernels/cuda/embedding.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace helix::cuda::kernels {
namespace {

constexpr int kWarmupLaunches = 10;

void BM_CudaEmbeddingLookup(benchmark::State& state) {
  if (!runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  set_device(0);
  const auto tokens = static_cast<Dimension>(state.range(0));
  const auto vocabulary = static_cast<Dimension>(state.range(1));
  const auto width = static_cast<Dimension>(state.range(2));
  std::vector<std::uint32_t> ids(static_cast<std::size_t>(tokens));
  for (std::size_t token = 0; token < ids.size(); ++token) {
    ids[token] =
        static_cast<std::uint32_t>((token * 7'919U + 17U) % static_cast<std::size_t>(vocabulary));
  }
  const Tensor token_ids_host = Tensor::from_values({tokens}, ids);
  const Tensor table_host = Tensor::random_uniform({vocabulary, width}, 1'212);
  const Tensor expected = helix::cpu::embedding_lookup(token_ids_host, table_host);
  const Tensor token_ids = token_ids_host.to(Device::cuda(0));
  const Tensor table = table_host.to(Device::cuda(0));
  Tensor output = Tensor::empty({tokens, width}, DType::kFloat32, Device::cuda(0));
  Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    embedding_lookup_out(token_ids, table, output, stream);
  }
  stream.synchronize();
  const auto output_matches_or_skip = [&state, &output, &expected]() {
    const Tensor actual = output.to(Device::cpu());
    for (std::size_t index = 0; index < expected.numel(); ++index) {
      if (actual.data_as<float>()[index] != expected.data_as<float>()[index]) {
        state.SkipWithError("CUDA embedding output does not match the CPU oracle");
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
    embedding_lookup_out(token_ids, table, output, stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }
  if (!output_matches_or_skip()) {
    return;
  }
  const auto elements = state.range(0) * state.range(2);
  state.SetItemsProcessed(state.iterations() * elements);
  state.SetBytesProcessed(state.iterations() *
                          (2 * elements * static_cast<std::int64_t>(sizeof(float)) +
                           state.range(0) * static_cast<std::int64_t>(sizeof(std::uint32_t))));
  state.counters["tokens"] = static_cast<double>(state.range(0));
  state.counters["vocabulary"] = static_cast<double>(state.range(1));
  state.counters["width"] = static_cast<double>(state.range(2));
}

BENCHMARK(BM_CudaEmbeddingLookup)
    ->Args({1, 128, 32})
    ->Args({7, 128, 32})
    ->Args({1, 4096, 256})
    ->Args({128, 4096, 256})
    ->Args({1, 32000, 1024})
    ->UseManualTime();

} // namespace
} // namespace helix::cuda::kernels
