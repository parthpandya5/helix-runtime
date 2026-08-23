#include "helix/attention/paged_reference.hpp"
#include "helix/attention/reference.hpp"
#include "helix/kv_cache/paged.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>

namespace helix::attention::cpu {
namespace {

void BM_CpuPagedAttention(benchmark::State& state, bool dimension_major) {
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
    if (dimension_major) {
      paged_scaled_dot_product_attention_dimension_major_out(query, view, query_position_offset,
                                                             output, workspace);
    } else {
      paged_scaled_dot_product_attention_out(query, view, query_position_offset, output, workspace);
    }
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

Tensor compact_paged_decode_attention(const Tensor& query, const Tensor& current_key,
                                      const Tensor& current_value,
                                      const kv_cache::PagedKvCache& cache,
                                      const kv_cache::PagedSequenceHandle& sequence,
                                      std::size_t history_tokens, std::size_t heads,
                                      std::size_t head_dimension) {
  Tensor combined_key =
      Tensor::empty({static_cast<Dimension>(heads), static_cast<Dimension>(history_tokens + 1),
                     static_cast<Dimension>(head_dimension)});
  Tensor combined_value = Tensor::empty(combined_key.shape());
  cache.copy_layer_prefix_out(sequence, 0, history_tokens, combined_key, combined_value);
  const std::size_t combined_head_elements = (history_tokens + 1) * head_dimension;
  for (std::size_t head = 0; head < heads; ++head) {
    const std::size_t destination = head * combined_head_elements + history_tokens * head_dimension;
    std::copy_n(current_key.data_as<float>() + head * head_dimension, head_dimension,
                combined_key.data_as<float>() + destination);
    std::copy_n(current_value.data_as<float>() + head * head_dimension, head_dimension,
                combined_value.data_as<float>() + destination);
  }
  return scaled_dot_product_attention(query, combined_key, combined_value, history_tokens);
}

void BM_CpuPagedDecodeAttention(benchmark::State& state, bool direct_block_table) {
  const auto heads = static_cast<std::size_t>(state.range(0));
  const auto head_dimension = static_cast<std::size_t>(state.range(1));
  const auto history_tokens = static_cast<std::size_t>(state.range(2));
  const auto block_tokens = static_cast<std::size_t>(state.range(3));
  const std::size_t physical_blocks =
      history_tokens / block_tokens + static_cast<std::size_t>(history_tokens % block_tokens != 0);
  kv_cache::PagedKvCache cache(
      kv_cache::PagedKvCacheConfig{1, heads, head_dimension, block_tokens, physical_blocks});
  const kv_cache::PagedSequenceHandle sequence = cache.create_sequence();
  const Tensor cached_key = Tensor::random_uniform(
      {1, static_cast<Dimension>(heads), static_cast<Dimension>(head_dimension)}, 95'101, -1.0F,
      1.0F);
  const Tensor cached_value = Tensor::random_uniform(cached_key.shape(), 95'102, -1.0F, 1.0F);
  for (std::size_t token = 0; token < history_tokens; ++token) {
    cache.append(sequence, cached_key, cached_value);
  }
  const Tensor query = Tensor::random_uniform(
      {static_cast<Dimension>(heads), 1, static_cast<Dimension>(head_dimension)}, 95'103, -1.0F,
      1.0F);
  const Tensor current_key = Tensor::random_uniform(query.shape(), 95'104, -1.0F, 1.0F);
  const Tensor current_value = Tensor::random_uniform(query.shape(), 95'105, -1.0F, 1.0F);

  const auto execute = [&]() {
    if (direct_block_table) {
      auto view = cache.read_layer(sequence, 0);
      return paged_scaled_dot_product_attention_with_tail(query, view, current_key, current_value);
    }
    return compact_paged_decode_attention(query, current_key, current_value, cache, sequence,
                                          history_tokens, heads, head_dimension);
  };
  {
    const Tensor warmup = execute();
    benchmark::DoNotOptimize(warmup.data());
  }

  for ([[maybe_unused]] auto iteration : state) {
    const Tensor output = execute();
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }

  const std::size_t avoided_temporary_bytes =
      2 * heads * (history_tokens + 1) * head_dimension * sizeof(float);
  state.counters["blocks"] = static_cast<double>(physical_blocks);
  state.counters["combined_kv_bytes"] =
      direct_block_table ? 0.0 : static_cast<double>(avoided_temporary_bytes);
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_CAPTURE(BM_CpuPagedAttention, DimensionMajor, true)
    ->Args({8, 1, 128, 64, 16})
    ->Args({8, 1, 1024, 64, 16})
    ->Args({3, 17, 29, 37, 7})
    ->Args({8, 128, 128, 64, 16});

BENCHMARK_CAPTURE(BM_CpuPagedAttention, DimensionTiled, false)
    ->Args({8, 1, 128, 64, 16})
    ->Args({8, 1, 1024, 64, 16})
    ->Args({3, 17, 29, 37, 7})
    ->Args({8, 128, 128, 64, 16});

BENCHMARK_CAPTURE(BM_CpuPagedDecodeAttention, CompactCopy, false)
    ->Args({4, 16, 3, 4})
    ->Args({4, 16, 15, 4})
    ->Args({8, 32, 63, 16})
    ->Args({8, 64, 255, 16})
    ->ArgNames({"heads", "dimension", "history", "block"});

BENCHMARK_CAPTURE(BM_CpuPagedDecodeAttention, DirectBlockTable, true)
    ->Args({4, 16, 3, 4})
    ->Args({4, 16, 15, 4})
    ->Args({8, 32, 63, 16})
    ->Args({8, 64, 255, 16})
    ->ArgNames({"heads", "dimension", "history", "block"});

} // namespace
} // namespace helix::attention::cpu
