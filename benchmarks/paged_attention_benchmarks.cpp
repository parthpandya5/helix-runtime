#include "helix/attention/paged_reference.hpp"
#include "helix/kv_cache/paged.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>

namespace helix::attention::cpu {
namespace {

void BM_CpuPagedAttention(benchmark::State& state) {
  const auto heads = static_cast<std::size_t>(state.range(0));
  const auto queries = static_cast<std::size_t>(state.range(1));
  const auto keys = static_cast<std::size_t>(state.range(2));
  const auto head_dimension = static_cast<std::size_t>(state.range(3));
  const auto block_tokens = static_cast<std::size_t>(state.range(4));
  const std::size_t physical_blocks =
      keys / block_tokens + static_cast<std::size_t>(keys % block_tokens != 0);
  kv_cache::PagedKvCache cache(
      kv_cache::PagedKvCacheConfig{1, heads, head_dimension, block_tokens, physical_blocks});
  const kv_cache::PagedSequenceHandle sequence = cache.create_sequence();
  const Tensor token_keys = Tensor::random_uniform(
      {1, static_cast<Dimension>(heads), static_cast<Dimension>(head_dimension)}, 95'001, -1.0F,
      1.0F);
  const Tensor token_values = Tensor::random_uniform(
      {1, static_cast<Dimension>(heads), static_cast<Dimension>(head_dimension)}, 95'002, -1.0F,
      1.0F);
  for (std::size_t token = 0; token < keys; ++token) {
    cache.append(sequence, token_keys, token_values);
  }

  const Tensor query =
      Tensor::random_uniform({static_cast<Dimension>(heads), static_cast<Dimension>(queries),
                              static_cast<Dimension>(head_dimension)},
                             95'003, -1.0F, 1.0F);
  Tensor output = Tensor::empty({static_cast<Dimension>(heads), static_cast<Dimension>(queries),
                                 static_cast<Dimension>(head_dimension)});
  Tensor workspace = Tensor::empty({static_cast<Dimension>(heads), static_cast<Dimension>(queries),
                                    static_cast<Dimension>(keys)});
  auto view = cache.read_layer(sequence, 0);
  const std::size_t query_position_offset = keys - queries;

  for ([[maybe_unused]] auto iteration : state) {
    paged_scaled_dot_product_attention_out(query, view, query_position_offset, output, workspace);
    benchmark::ClobberMemory();
  }

  const std::int64_t dot_flops =
      4 * state.range(0) * state.range(1) * state.range(2) * state.range(3);
  state.counters["blocks"] = static_cast<double>(physical_blocks);
  state.counters["dot_FLOP/s"] = benchmark::Counter(static_cast<double>(dot_flops),
                                                    benchmark::Counter::kIsIterationInvariantRate);
  state.counters["workspace_bytes"] = static_cast<double>(workspace.nbytes());
  state.SetItemsProcessed(state.iterations() * state.range(0) * state.range(1));
}

BENCHMARK(BM_CpuPagedAttention)
    ->Args({8, 1, 128, 64, 16})
    ->Args({8, 1, 1024, 64, 16})
    ->Args({3, 17, 29, 37, 7})
    ->Args({8, 128, 128, 64, 16});

} // namespace
} // namespace helix::attention::cpu
