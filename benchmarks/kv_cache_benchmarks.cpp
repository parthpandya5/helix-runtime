#include "helix/kv_cache/contiguous.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace helix::kv_cache {
namespace {

void BM_ContiguousKvLifecycle(benchmark::State& state) {
  const auto layers = static_cast<std::size_t>(state.range(0));
  const auto heads = static_cast<std::size_t>(state.range(1));
  const auto head_dimension = static_cast<std::size_t>(state.range(2));
  const auto capacity = static_cast<std::size_t>(state.range(3));
  const auto appended_tokens = static_cast<std::size_t>(state.range(4));
  ContiguousKvCache cache(ContiguousKvCacheConfig{layers, heads, head_dimension,
                                                  std::numeric_limits<std::size_t>::max()});
  const Tensor keys =
      Tensor::random_uniform({static_cast<Dimension>(layers), static_cast<Dimension>(heads),
                              static_cast<Dimension>(head_dimension)},
                             81'001, -1.0F, 1.0F);
  const Tensor values =
      Tensor::random_uniform({static_cast<Dimension>(layers), static_cast<Dimension>(heads),
                              static_cast<Dimension>(head_dimension)},
                             81'002, -1.0F, 1.0F);

  for ([[maybe_unused]] auto iteration : state) {
    const SequenceHandle sequence = cache.create_sequence(capacity);
    for (std::size_t token = 0; token < appended_tokens; ++token) {
      cache.append(sequence, keys, values);
    }
    benchmark::DoNotOptimize(cache.length(sequence));
    cache.release_sequence(sequence);
  }

  state.counters["capacity_bytes"] = static_cast<double>(cache.sequence_bytes(capacity));
  state.counters["tokens_per_trace"] = static_cast<double>(appended_tokens);
  state.SetItemsProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(appended_tokens));
}

void BM_ContiguousKvLayerCopy(benchmark::State& state) {
  const auto layers = static_cast<std::size_t>(state.range(0));
  const auto heads = static_cast<std::size_t>(state.range(1));
  const auto head_dimension = static_cast<std::size_t>(state.range(2));
  const auto tokens = static_cast<std::size_t>(state.range(3));
  ContiguousKvCache cache(ContiguousKvCacheConfig{layers, heads, head_dimension,
                                                  std::numeric_limits<std::size_t>::max()});
  const SequenceHandle sequence = cache.create_sequence(tokens);
  const Tensor keys =
      Tensor::random_uniform({static_cast<Dimension>(layers), static_cast<Dimension>(heads),
                              static_cast<Dimension>(head_dimension)},
                             82'001, -1.0F, 1.0F);
  const Tensor values =
      Tensor::random_uniform({static_cast<Dimension>(layers), static_cast<Dimension>(heads),
                              static_cast<Dimension>(head_dimension)},
                             82'002, -1.0F, 1.0F);
  for (std::size_t token = 0; token < tokens; ++token) {
    cache.append(sequence, keys, values);
  }
  Tensor key_output = Tensor::empty({static_cast<Dimension>(heads), static_cast<Dimension>(tokens),
                                     static_cast<Dimension>(head_dimension)});
  Tensor value_output = Tensor::empty(key_output.shape());

  for ([[maybe_unused]] auto iteration : state) {
    cache.copy_layer_out(sequence, layers - 1, key_output, value_output);
    benchmark::ClobberMemory();
  }

  const std::size_t copied_bytes = key_output.nbytes() + value_output.nbytes();
  state.counters["copied_bytes"] = static_cast<double>(copied_bytes);
  state.SetBytesProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(copied_bytes));
}

BENCHMARK(BM_ContiguousKvLifecycle)
    ->Args({2, 2, 64, 128, 128})
    ->Args({4, 8, 64, 512, 32})
    ->Args({3, 3, 37, 29, 29});

BENCHMARK(BM_ContiguousKvLayerCopy)
    ->Args({2, 2, 64, 128})
    ->Args({4, 8, 64, 512})
    ->Args({3, 3, 37, 29});

} // namespace
} // namespace helix::kv_cache
