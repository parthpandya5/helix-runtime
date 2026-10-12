#include "helix/kv_cache/cuda_paged.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <benchmark/benchmark.h>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace helix::kv_cache {
namespace {

constexpr int kWarmupIterations = 10;

bool verify_appended_token(benchmark::State& state, CudaPagedKvCache& cache,
                           const CudaPagedSequenceHandle& sequence, const Tensor& expected_keys,
                           const Tensor& expected_values) {
  auto view = cache.read_sequence(sequence);
  if (view.length() != 1 || view.physical_blocks().size() != 1) {
    state.SkipWithError("CUDA paged KV benchmark produced invalid sequence metadata");
    return false;
  }
  const std::size_t physical_block = view.physical_blocks().front();
  const PagedKvCacheConfig& config = cache.config();
  const Tensor host_keys = expected_keys.to(Device::cpu());
  const Tensor host_values = expected_values.to(Device::cpu());
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    const Tensor packed_keys = view.layer_key_blocks()[layer].to(Device::cpu());
    const Tensor packed_values = view.layer_value_blocks()[layer].to(Device::cpu());
    for (std::size_t head = 0; head < config.kv_heads; ++head) {
      for (std::size_t coordinate = 0; coordinate < config.head_dimension; ++coordinate) {
        const std::size_t source =
            (layer * config.kv_heads + head) * config.head_dimension + coordinate;
        const std::size_t destination =
            (((physical_block * config.kv_heads + head) * config.block_tokens) *
             config.head_dimension) +
            coordinate;
        if (packed_keys.data_as<float>()[destination] != host_keys.data_as<float>()[source] ||
            packed_values.data_as<float>()[destination] != host_values.data_as<float>()[source]) {
          state.SkipWithError("CUDA paged KV benchmark commit does not match its input oracle");
          return false;
        }
      }
    }
  }
  return true;
}

void BM_CudaPagedKvSynchronousAppend(benchmark::State& state) {
  if (!helix::cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  helix::cuda::set_device(0);
  const PagedKvCacheConfig config{
      static_cast<std::size_t>(state.range(0)), static_cast<std::size_t>(state.range(1)),
      static_cast<std::size_t>(state.range(2)), static_cast<std::size_t>(state.range(3)), 2};
  CudaPagedKvCache cache(config);
  const Shape token_shape{static_cast<Dimension>(config.layers),
                          static_cast<Dimension>(config.kv_heads),
                          static_cast<Dimension>(config.head_dimension)};
  const Tensor token_keys =
      Tensor::random_uniform(token_shape, 42, -1.0F, 1.0F, DType::kFloat32, Device::cuda(0));
  const Tensor token_values =
      Tensor::random_uniform(token_shape, 43, -1.0F, 1.0F, DType::kFloat32, Device::cuda(0));
  helix::cuda::Stream stream;

  auto append_then_release = [&] {
    const CudaPagedSequenceHandle sequence = cache.create_sequence();
    cache.append(sequence, token_keys, token_values, stream);
    cache.release_sequence(sequence);
  };
  for (int warmup = 0; warmup < kWarmupIterations; ++warmup) {
    append_then_release();
  }

  auto verify_once = [&] {
    const CudaPagedSequenceHandle sequence = cache.create_sequence();
    cache.append(sequence, token_keys, token_values, stream);
    const bool correct = verify_appended_token(state, cache, sequence, token_keys, token_values);
    cache.release_sequence(sequence);
    return correct;
  };
  if (!verify_once()) {
    return;
  }

  for ([[maybe_unused]] auto iteration : state) {
    const CudaPagedSequenceHandle sequence = cache.create_sequence();
    const auto start = std::chrono::steady_clock::now();
    cache.append(sequence, token_keys, token_values, stream);
    const auto stop = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(stop - start).count());
    cache.release_sequence(sequence);
  }

  if (!verify_once()) {
    return;
  }
  const PagedKvCacheStats stats = cache.stats();
  if (stats.active_sequence_count != 0 || stats.used_blocks != 0 ||
      stats.bytes_free != stats.bytes_reserved) {
    state.SkipWithError("CUDA paged KV benchmark leaked sequence or block ownership");
    return;
  }

  const std::size_t copied_bytes = cache.bytes_per_token();
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(copied_bytes));
  state.SetItemsProcessed(state.iterations());
  state.counters["block_tokens"] = static_cast<double>(config.block_tokens);
  state.counters["d2d_copies"] = static_cast<double>(2 * config.layers * config.kv_heads);
  state.counters["physical_pool_bytes"] = static_cast<double>(stats.bytes_reserved);
  state.counters["synchronizations"] = 1.0;
  state.counters["token_bytes"] = static_cast<double>(copied_bytes);
}

BENCHMARK(BM_CudaPagedKvSynchronousAppend)
    ->Args({1, 1, 32, 8})
    ->Args({2, 4, 16, 8})
    ->Args({16, 32, 128, 16})
    ->ArgNames({"layers", "heads", "head_dimension", "block_tokens"})
    ->UseManualTime();

} // namespace
} // namespace helix::kv_cache
