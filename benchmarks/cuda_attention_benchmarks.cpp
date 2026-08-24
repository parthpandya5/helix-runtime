#include "helix/attention/cuda.hpp"
#include "helix/attention/cuda_paged.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace helix::attention::cuda {
namespace {

constexpr int kWarmupLaunches = 10;

void BM_CudaContiguousAttention(benchmark::State& state) {
  if (!helix::cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  helix::cuda::set_device(0);
  const auto heads = static_cast<Dimension>(state.range(0));
  const auto queries = static_cast<Dimension>(state.range(1));
  const auto keys = static_cast<Dimension>(state.range(2));
  const auto head_dimension = static_cast<Dimension>(state.range(3));
  const auto value_dimension = static_cast<Dimension>(state.range(4));
  const Tensor query = Tensor::random_uniform({heads, queries, head_dimension}, 9'001, -1.0F, 1.0F,
                                              DType::kFloat32, Device::cuda(0));
  const Tensor key = Tensor::random_uniform({heads, keys, head_dimension}, 9'002, -1.0F, 1.0F,
                                            DType::kFloat32, Device::cuda(0));
  const Tensor value = Tensor::random_uniform({heads, keys, value_dimension}, 9'003, -1.0F, 1.0F,
                                              DType::kFloat32, Device::cuda(0));
  Tensor output =
      Tensor::empty({heads, queries, value_dimension}, DType::kFloat32, Device::cuda(0));
  Tensor workspace = Tensor::empty({heads, queries, keys}, DType::kFloat32, Device::cuda(0));
  const auto query_position_offset = static_cast<std::size_t>(keys - queries);
  helix::cuda::Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    scaled_dot_product_attention_out(query, key, value, query_position_offset, output, workspace,
                                     stream);
  }
  stream.synchronize();

  helix::cuda::Event start;
  helix::cuda::Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    scaled_dot_product_attention_out(query, key, value, query_position_offset, output, workspace,
                                     stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(helix::cuda::elapsed_time_ms(start, stop)) / 1000.0);
  }

  const std::int64_t dot_flops =
      2 * state.range(0) * state.range(1) * state.range(2) * (state.range(3) + state.range(4));
  state.counters["dot_FLOP/s"] = benchmark::Counter(static_cast<double>(dot_flops),
                                                    benchmark::Counter::kIsIterationInvariantRate);
  state.counters["workspace_bytes"] = static_cast<double>(workspace.nbytes());
  state.SetItemsProcessed(state.iterations() * state.range(0) * state.range(1));
}

void BM_CudaPagedAttention(benchmark::State& state) {
  if (!helix::cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  helix::cuda::set_device(0);
  const auto heads = static_cast<Dimension>(state.range(0));
  const auto queries = static_cast<Dimension>(state.range(1));
  const auto keys = static_cast<Dimension>(state.range(2));
  const auto head_dimension = static_cast<Dimension>(state.range(3));
  const auto value_dimension = static_cast<Dimension>(state.range(4));
  const auto block_tokens = static_cast<Dimension>(state.range(5));
  const auto kv_tokens = static_cast<std::size_t>(keys);
  const auto tokens_per_block = static_cast<std::size_t>(block_tokens);
  const std::size_t logical_blocks = 1 + (kv_tokens - 1) / tokens_per_block;
  const std::size_t physical_blocks = logical_blocks + 2;
  std::vector<std::uint32_t> block_ids(logical_blocks);
  for (std::size_t logical_block = 0; logical_block < logical_blocks; ++logical_block) {
    block_ids[logical_block] = static_cast<std::uint32_t>(physical_blocks - 1 - logical_block);
  }

  const Tensor query = Tensor::random_uniform({heads, queries, head_dimension}, 9'101, -1.0F, 1.0F,
                                              DType::kFloat32, Device::cuda(0));
  const Tensor key_blocks = Tensor::random_uniform(
      {static_cast<Dimension>(physical_blocks), heads, block_tokens, head_dimension}, 9'102, -1.0F,
      1.0F, DType::kFloat32, Device::cuda(0));
  const Tensor value_blocks = Tensor::random_uniform(
      {static_cast<Dimension>(physical_blocks), heads, block_tokens, value_dimension}, 9'103, -1.0F,
      1.0F, DType::kFloat32, Device::cuda(0));
  PagedBlockTable block_table(block_ids, physical_blocks, 0);
  Tensor output =
      Tensor::empty({heads, queries, value_dimension}, DType::kFloat32, Device::cuda(0));
  Tensor workspace = Tensor::empty({heads, queries, keys}, DType::kFloat32, Device::cuda(0));
  const auto query_position_offset = static_cast<std::size_t>(keys - queries);
  helix::cuda::Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    paged_scaled_dot_product_attention_out(query, key_blocks, value_blocks, block_table, kv_tokens,
                                           query_position_offset, output, workspace, stream);
  }
  stream.synchronize();

  helix::cuda::Event start;
  helix::cuda::Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    paged_scaled_dot_product_attention_out(query, key_blocks, value_blocks, block_table, kv_tokens,
                                           query_position_offset, output, workspace, stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(helix::cuda::elapsed_time_ms(start, stop)) / 1000.0);
  }

  const std::int64_t dot_flops =
      2 * state.range(0) * state.range(1) * state.range(2) * (state.range(3) + state.range(4));
  state.counters["dot_FLOP/s"] = benchmark::Counter(static_cast<double>(dot_flops),
                                                    benchmark::Counter::kIsIterationInvariantRate);
  state.counters["workspace_bytes"] = static_cast<double>(workspace.nbytes());
  state.counters["block_table_bytes"] = static_cast<double>(block_table.size_bytes());
  state.counters["active_kv_bytes"] = static_cast<double>(heads) * static_cast<double>(keys) *
                                      static_cast<double>(head_dimension + value_dimension) *
                                      static_cast<double>(sizeof(float));
  state.counters["physical_kv_bytes"] =
      static_cast<double>(physical_blocks) * static_cast<double>(heads) *
      static_cast<double>(block_tokens) * static_cast<double>(head_dimension + value_dimension) *
      static_cast<double>(sizeof(float));
  state.SetItemsProcessed(state.iterations() * state.range(0) * state.range(1));
}

BENCHMARK(BM_CudaContiguousAttention)
    ->Args({8, 1, 128, 64, 64})
    ->Args({8, 1, 1024, 64, 64})
    ->Args({3, 17, 29, 37, 41})
    ->Args({8, 128, 128, 64, 64})
    ->UseManualTime();

BENCHMARK(BM_CudaPagedAttention)
    ->Args({8, 1, 128, 64, 64, 16})
    ->Args({8, 1, 1024, 64, 64, 16})
    ->Args({3, 17, 29, 37, 41, 8})
    ->Args({8, 128, 128, 64, 64, 16})
    ->UseManualTime();

} // namespace
} // namespace helix::attention::cuda
