#include "helix/tensor/tensor.hpp"

#include <benchmark/benchmark.h>
#include <cstdint>

namespace helix {
namespace {

void BM_TensorZeros(benchmark::State& state) {
  const auto elements = static_cast<Dimension>(state.range(0));
  for ([[maybe_unused]] auto iteration : state) {
    Tensor tensor = Tensor::zeros({elements});
    benchmark::DoNotOptimize(tensor.data());
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) *
                          static_cast<std::int64_t>(sizeof(float)));
}

BENCHMARK(BM_TensorZeros)->Arg(1)->Arg(1024)->Arg(1 << 20);

} // namespace
} // namespace helix
