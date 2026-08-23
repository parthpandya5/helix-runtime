#include "helix/kv_cache/contiguous.hpp"
#include "helix/kv_cache/paged.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

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

void copy_current_token(const Tensor& current_keys, const Tensor& current_values,
                        std::size_t history_tokens, std::size_t heads, std::size_t head_dimension,
                        Tensor& combined_keys, Tensor& combined_values) {
  const float* current_key_values = current_keys.data_as<float>();
  const float* current_value_values = current_values.data_as<float>();
  float* combined_key_values = combined_keys.data_as<float>();
  float* combined_value_values = combined_values.data_as<float>();
  const std::size_t combined_head_elements = (history_tokens + 1) * head_dimension;
  for (std::size_t head = 0; head < heads; ++head) {
    const std::size_t destination_offset =
        head * combined_head_elements + history_tokens * head_dimension;
    std::copy_n(current_key_values + head * head_dimension, head_dimension,
                combined_key_values + destination_offset);
    std::copy_n(current_value_values + head * head_dimension, head_dimension,
                combined_value_values + destination_offset);
  }
}

struct DecodeStagingBuffers {
  Tensor keys;
  Tensor values;
};

DecodeStagingBuffers stage_decode_history(const ContiguousKvCache& cache,
                                          const SequenceHandle& sequence, const Tensor& token_keys,
                                          const Tensor& token_values, std::size_t history_tokens,
                                          std::size_t heads, std::size_t head_dimension,
                                          bool direct_prefix_copy) {
  Tensor combined_keys =
      Tensor::empty({static_cast<Dimension>(heads), static_cast<Dimension>(history_tokens + 1),
                     static_cast<Dimension>(head_dimension)});
  Tensor combined_values = Tensor::empty(combined_keys.shape());
  if (direct_prefix_copy) {
    cache.copy_layer_prefix_out(sequence, 0, history_tokens, combined_keys, combined_values);
  } else {
    const KvLayer history = cache.retrieve_layer(sequence, 0);
    const std::size_t history_head_elements = history_tokens * head_dimension;
    const std::size_t combined_head_elements = (history_tokens + 1) * head_dimension;
    for (std::size_t head = 0; head < heads; ++head) {
      std::copy_n(history.keys.data_as<float>() + head * history_head_elements,
                  history_head_elements,
                  combined_keys.data_as<float>() + head * combined_head_elements);
      std::copy_n(history.values.data_as<float>() + head * history_head_elements,
                  history_head_elements,
                  combined_values.data_as<float>() + head * combined_head_elements);
    }
  }
  copy_current_token(token_keys, token_values, history_tokens, heads, head_dimension, combined_keys,
                     combined_values);
  return DecodeStagingBuffers{std::move(combined_keys), std::move(combined_values)};
}

void BM_ContiguousKvDecodeStaging(benchmark::State& state, bool direct_prefix_copy) {
  const auto heads = static_cast<std::size_t>(state.range(0));
  const auto head_dimension = static_cast<std::size_t>(state.range(1));
  const auto history_tokens = static_cast<std::size_t>(state.range(2));
  ContiguousKvCache cache(
      ContiguousKvCacheConfig{1, heads, head_dimension, std::numeric_limits<std::size_t>::max()});
  const SequenceHandle sequence = cache.create_sequence(history_tokens);
  const Tensor token_keys = Tensor::random_uniform(
      {1, static_cast<Dimension>(heads), static_cast<Dimension>(head_dimension)}, 82'101, -1.0F,
      1.0F);
  const Tensor token_values = Tensor::random_uniform(token_keys.shape(), 82'102, -1.0F, 1.0F);
  for (std::size_t token = 0; token < history_tokens; ++token) {
    cache.append(sequence, token_keys, token_values);
  }

  const DecodeStagingBuffers warmup =
      stage_decode_history(cache, sequence, token_keys, token_values, history_tokens, heads,
                           head_dimension, direct_prefix_copy);
  benchmark::DoNotOptimize(warmup.keys.data());
  benchmark::DoNotOptimize(warmup.values.data());

  for ([[maybe_unused]] auto iteration : state) {
    const DecodeStagingBuffers staged =
        stage_decode_history(cache, sequence, token_keys, token_values, history_tokens, heads,
                             head_dimension, direct_prefix_copy);
    benchmark::DoNotOptimize(staged.keys.data());
    benchmark::DoNotOptimize(staged.values.data());
    benchmark::ClobberMemory();
  }

  const std::size_t history_bytes = 2 * heads * history_tokens * head_dimension * sizeof(float);
  state.counters["history_bytes"] = static_cast<double>(history_bytes);
  state.counters["history_copy_passes"] = direct_prefix_copy ? 1.0 : 2.0;
  state.counters["temporary_tensors"] = direct_prefix_copy ? 2.0 : 4.0;
}

void BM_PagedKvLifecycle(benchmark::State& state) {
  const auto layers = static_cast<std::size_t>(state.range(0));
  const auto heads = static_cast<std::size_t>(state.range(1));
  const auto head_dimension = static_cast<std::size_t>(state.range(2));
  const auto block_tokens = static_cast<std::size_t>(state.range(3));
  const auto appended_tokens = static_cast<std::size_t>(state.range(4));
  const std::size_t physical_blocks = appended_tokens / block_tokens +
                                      static_cast<std::size_t>(appended_tokens % block_tokens != 0);
  PagedKvCache cache(
      PagedKvCacheConfig{layers, heads, head_dimension, block_tokens, physical_blocks});
  const Tensor keys =
      Tensor::random_uniform({static_cast<Dimension>(layers), static_cast<Dimension>(heads),
                              static_cast<Dimension>(head_dimension)},
                             83'001, -1.0F, 1.0F);
  const Tensor values =
      Tensor::random_uniform({static_cast<Dimension>(layers), static_cast<Dimension>(heads),
                              static_cast<Dimension>(head_dimension)},
                             83'002, -1.0F, 1.0F);

  for ([[maybe_unused]] auto iteration : state) {
    const PagedSequenceHandle sequence = cache.create_sequence();
    for (std::size_t token = 0; token < appended_tokens; ++token) {
      cache.append(sequence, keys, values);
    }
    benchmark::DoNotOptimize(cache.length(sequence));
    cache.release_sequence(sequence);
  }

  state.counters["blocks_per_trace"] = static_cast<double>(physical_blocks);
  state.counters["pool_bytes"] = static_cast<double>(cache.stats().bytes_reserved);
  state.counters["tail_slots"] =
      static_cast<double>(physical_blocks * block_tokens - appended_tokens);
  state.SetItemsProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(appended_tokens));
}

void BM_PagedKvLayerCopy(benchmark::State& state) {
  const auto layers = static_cast<std::size_t>(state.range(0));
  const auto heads = static_cast<std::size_t>(state.range(1));
  const auto head_dimension = static_cast<std::size_t>(state.range(2));
  const auto block_tokens = static_cast<std::size_t>(state.range(3));
  const auto tokens = static_cast<std::size_t>(state.range(4));
  const std::size_t physical_blocks =
      tokens / block_tokens + static_cast<std::size_t>(tokens % block_tokens != 0);
  PagedKvCache cache(
      PagedKvCacheConfig{layers, heads, head_dimension, block_tokens, physical_blocks});
  const PagedSequenceHandle sequence = cache.create_sequence();
  const Tensor keys =
      Tensor::random_uniform({static_cast<Dimension>(layers), static_cast<Dimension>(heads),
                              static_cast<Dimension>(head_dimension)},
                             84'001, -1.0F, 1.0F);
  const Tensor values =
      Tensor::random_uniform({static_cast<Dimension>(layers), static_cast<Dimension>(heads),
                              static_cast<Dimension>(head_dimension)},
                             84'002, -1.0F, 1.0F);
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
  state.counters["blocks"] = static_cast<double>(physical_blocks);
  state.counters["copied_bytes"] = static_cast<double>(copied_bytes);
  state.SetBytesProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(copied_bytes));
}

void BM_PagedKvPrefixForkLifecycle(benchmark::State& state, bool share_blocks) {
  const auto layers = static_cast<std::size_t>(state.range(0));
  const auto heads = static_cast<std::size_t>(state.range(1));
  const auto head_dimension = static_cast<std::size_t>(state.range(2));
  const auto block_tokens = static_cast<std::size_t>(state.range(3));
  const auto prefix_tokens = static_cast<std::size_t>(state.range(4));
  if (block_tokens == 0 || prefix_tokens == 0 || prefix_tokens % block_tokens != 0) {
    state.SkipWithError("prefix benchmark requires a positive, block-aligned prefix");
    return;
  }
  const std::size_t prefix_blocks = prefix_tokens / block_tokens;
  PagedKvCache cache(
      PagedKvCacheConfig{layers, heads, head_dimension, block_tokens, 2 * prefix_blocks});

  const Shape prefix_shape{static_cast<Dimension>(heads), static_cast<Dimension>(prefix_tokens),
                           static_cast<Dimension>(head_dimension)};
  std::vector<Tensor> layer_keys;
  std::vector<Tensor> layer_values;
  layer_keys.reserve(layers);
  layer_values.reserve(layers);
  for (std::size_t layer = 0; layer < layers; ++layer) {
    layer_keys.push_back(Tensor::random_uniform(prefix_shape, 85'001 + layer, -1.0F, 1.0F));
    layer_values.push_back(Tensor::random_uniform(prefix_shape, 86'001 + layer, -1.0F, 1.0F));
  }
  const PagedSequenceHandle source = cache.create_sequence();
  cache.append_layer_major(source, layer_keys, layer_values);

  const auto create_child = [&cache, &layer_keys, &layer_values, source, prefix_tokens,
                             share_blocks] {
    if (share_blocks) {
      return cache.fork_sequence(source, prefix_tokens);
    }
    const PagedSequenceHandle child = cache.create_sequence();
    cache.append_layer_major(child, layer_keys, layer_values);
    return child;
  };

  const PagedSequenceHandle warmup = create_child();
  benchmark::DoNotOptimize(cache.length(warmup));
  cache.release_sequence(warmup);
  for ([[maybe_unused]] auto iteration : state) {
    const PagedSequenceHandle child = create_child();
    benchmark::DoNotOptimize(cache.length(child));
    cache.release_sequence(child);
  }

  const std::size_t logical_payload_bytes = prefix_tokens * cache.bytes_per_token();
  state.counters["additional_assigned_bytes"] =
      static_cast<double>(share_blocks ? 0 : logical_payload_bytes);
  state.counters["block_table_entries"] = static_cast<double>(prefix_blocks);
  state.counters["logical_payload_bytes"] = static_cast<double>(logical_payload_bytes);
  state.counters["payload_copy_bytes"] =
      static_cast<double>(share_blocks ? 0 : logical_payload_bytes);
  state.SetItemsProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(prefix_tokens));
}

BENCHMARK(BM_ContiguousKvLifecycle)
    ->Args({2, 2, 64, 128, 128})
    ->Args({4, 8, 64, 512, 32})
    ->Args({3, 3, 37, 29, 29});

BENCHMARK(BM_ContiguousKvLayerCopy)
    ->Args({2, 2, 64, 128})
    ->Args({4, 8, 64, 512})
    ->Args({3, 3, 37, 29});

BENCHMARK_CAPTURE(BM_ContiguousKvDecodeStaging, TwoStageSnapshot, false)
    ->Args({4, 8, 7})
    ->Args({4, 16, 31})
    ->Args({8, 16, 63})
    ->ArgNames({"heads", "head_dimension", "history_tokens"});

BENCHMARK_CAPTURE(BM_ContiguousKvDecodeStaging, DirectPrefixCopy, true)
    ->Args({4, 8, 7})
    ->Args({4, 16, 31})
    ->Args({8, 16, 63})
    ->ArgNames({"heads", "head_dimension", "history_tokens"});

BENCHMARK(BM_PagedKvLifecycle)
    ->Args({2, 2, 64, 16, 128})
    ->Args({4, 8, 64, 16, 32})
    ->Args({3, 3, 37, 7, 29});

BENCHMARK(BM_PagedKvLayerCopy)
    ->Args({2, 2, 64, 16, 128})
    ->Args({4, 8, 64, 16, 512})
    ->Args({3, 3, 37, 7, 29});

BENCHMARK_CAPTURE(BM_PagedKvPrefixForkLifecycle, DuplicatePayload, false)
    ->Args({2, 2, 32, 8, 32})
    ->Args({3, 3, 37, 7, 35})
    ->Args({4, 4, 64, 16, 128})
    ->ArgNames({"layers", "heads", "head_dimension", "block_tokens", "prefix_tokens"});

BENCHMARK_CAPTURE(BM_PagedKvPrefixForkLifecycle, SharedBlocks, true)
    ->Args({2, 2, 32, 8, 32})
    ->Args({3, 3, 37, 7, 35})
    ->Args({4, 4, 64, 16, 128})
    ->ArgNames({"layers", "heads", "head_dimension", "block_tokens", "prefix_tokens"});

} // namespace
} // namespace helix::kv_cache
