#include "helix/attention/reference.hpp"

#include <benchmark/benchmark.h>
#include <cstdint>

namespace helix::attention::cpu {
namespace {

void BM_CpuContiguousAttention(benchmark::State& state) {
  const auto heads = static_cast<Dimension>(state.range(0));
  const auto queries = static_cast<Dimension>(state.range(1));
  const auto keys = static_cast<Dimension>(state.range(2));
  const auto head_dimension = static_cast<Dimension>(state.range(3));
  const auto value_dimension = static_cast<Dimension>(state.range(4));
  const Tensor query = Tensor::random_uniform({heads, queries, head_dimension}, 7'001, -1.0F, 1.0F);
  const Tensor key = Tensor::random_uniform({heads, keys, head_dimension}, 7'002, -1.0F, 1.0F);
  const Tensor value = Tensor::random_uniform({heads, keys, value_dimension}, 7'003, -1.0F, 1.0F);
  Tensor output = Tensor::empty({heads, queries, value_dimension});
  Tensor workspace = Tensor::empty({heads, queries, keys});
  const auto query_position_offset = static_cast<std::size_t>(keys - queries);

  for ([[maybe_unused]] auto iteration : state) {
    scaled_dot_product_attention_out(query, key, value, query_position_offset, output, workspace);
    benchmark::ClobberMemory();
  }

  const std::int64_t dot_flops =
      2 * state.range(0) * state.range(1) * state.range(2) * (state.range(3) + state.range(4));
  state.counters["dot_FLOP/s"] = benchmark::Counter(static_cast<double>(dot_flops),
                                                    benchmark::Counter::kIsIterationInvariantRate);
  state.counters["workspace_bytes"] = static_cast<double>(workspace.nbytes());
  state.SetItemsProcessed(state.iterations() * state.range(0) * state.range(1));
}

BENCHMARK(BM_CpuContiguousAttention)
    ->Args({8, 1, 128, 64, 64})
    ->Args({8, 1, 1024, 64, 64})
    ->Args({3, 17, 29, 37, 37})
    ->Args({3, 17, 29, 37, 41})
    ->Args({8, 128, 128, 64, 64});

} // namespace
} // namespace helix::attention::cpu
