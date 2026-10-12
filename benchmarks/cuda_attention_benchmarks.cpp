#include "helix/attention/cuda.hpp"
#include "helix/attention/cuda_paged.hpp"
#include "helix/attention/reference.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace helix::attention::cuda {
namespace {

constexpr int kWarmupLaunches = 10;

Tensor pack_logical_blocks(const Tensor& logical, std::span<const std::uint32_t> block_table,
                           Dimension physical_blocks, Dimension block_tokens, float padding_value) {
  const auto heads = static_cast<std::size_t>(logical.shape()[0]);
  const auto tokens = static_cast<std::size_t>(logical.shape()[1]);
  const auto width = static_cast<std::size_t>(logical.shape()[2]);
  const auto physical_count = static_cast<std::size_t>(physical_blocks);
  const auto tokens_per_block = static_cast<std::size_t>(block_tokens);
  std::vector<float> packed(physical_count * heads * tokens_per_block * width, padding_value);
  const auto* source = logical.data_as<float>();
  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t token = 0; token < tokens; ++token) {
      const std::size_t logical_block = token / tokens_per_block;
      const std::size_t token_in_block = token % tokens_per_block;
      const std::size_t physical_block = static_cast<std::size_t>(block_table[logical_block]);
      for (std::size_t dimension = 0; dimension < width; ++dimension) {
        const std::size_t source_index = (head * tokens + token) * width + dimension;
        const std::size_t destination_index =
            (((physical_block * heads + head) * tokens_per_block + token_in_block) * width) +
            dimension;
        packed[destination_index] = source[source_index];
      }
    }
  }
  return Tensor::from_values(
      {physical_blocks, logical.shape()[0], block_tokens, logical.shape()[2]}, packed);
}

Tensor concatenate_token_axis(const Tensor& prefix, const Tensor& tail) {
  const auto heads = static_cast<std::size_t>(prefix.shape()[0]);
  const auto prefix_tokens = static_cast<std::size_t>(prefix.shape()[1]);
  const auto tail_tokens = static_cast<std::size_t>(tail.shape()[1]);
  const auto width = static_cast<std::size_t>(prefix.shape()[2]);
  Tensor result =
      Tensor::empty({prefix.shape()[0], prefix.shape()[1] + tail.shape()[1], prefix.shape()[2]});
  for (std::size_t head = 0; head < heads; ++head) {
    float* destination = result.data_as<float>() + head * (prefix_tokens + tail_tokens) * width;
    if (prefix_tokens != 0) {
      std::copy_n(prefix.data_as<float>() + head * prefix_tokens * width, prefix_tokens * width,
                  destination);
    }
    std::copy_n(tail.data_as<float>() + head * tail_tokens * width, tail_tokens * width,
                destination + prefix_tokens * width);
  }
  return result;
}

bool all_close(const Tensor& actual_device, const Tensor& expected_host, float tolerance) {
  const Tensor actual = actual_device.to(Device::cpu());
  if (actual.shape() != expected_host.shape()) {
    return false;
  }
  for (std::size_t index = 0; index < actual.numel(); ++index) {
    const float actual_value = actual.data_as<float>()[index];
    const float expected_value = expected_host.data_as<float>()[index];
    if (!std::isfinite(actual_value) || !std::isfinite(expected_value) ||
        std::abs(actual_value - expected_value) > tolerance) {
      return false;
    }
  }
  return true;
}

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

void BM_CudaPagedAttentionWithTail(benchmark::State& state) {
  if (!helix::cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  helix::cuda::set_device(0);
  const auto heads = static_cast<Dimension>(state.range(0));
  const auto history_tokens = static_cast<Dimension>(state.range(1));
  const auto tail_tokens = static_cast<Dimension>(state.range(2));
  const auto head_dimension = static_cast<Dimension>(state.range(3));
  const auto value_dimension = static_cast<Dimension>(state.range(4));
  const auto block_tokens = static_cast<Dimension>(state.range(5));
  const auto history_count = static_cast<std::size_t>(history_tokens);
  const auto tokens_per_block = static_cast<std::size_t>(block_tokens);
  const std::size_t logical_blocks =
      history_count == 0 ? 0 : 1 + (history_count - 1) / tokens_per_block;
  const std::size_t physical_blocks = logical_blocks + 2;
  std::vector<std::uint32_t> block_ids(logical_blocks);
  for (std::size_t logical_block = 0; logical_block < logical_blocks; ++logical_block) {
    block_ids[logical_block] = static_cast<std::uint32_t>(physical_blocks - 1 - logical_block);
  }

  const Tensor query_host =
      Tensor::random_uniform({heads, tail_tokens, head_dimension}, 9'201, -1.0F, 1.0F);
  const Tensor history_key_host =
      Tensor::random_uniform({heads, history_tokens, head_dimension}, 9'202, -1.0F, 1.0F);
  const Tensor history_value_host =
      Tensor::random_uniform({heads, history_tokens, value_dimension}, 9'203, -1.0F, 1.0F);
  const Tensor tail_key_host =
      Tensor::random_uniform({heads, tail_tokens, head_dimension}, 9'204, -1.0F, 1.0F);
  const Tensor tail_value_host =
      Tensor::random_uniform({heads, tail_tokens, value_dimension}, 9'205, -1.0F, 1.0F);
  const Tensor combined_key = concatenate_token_axis(history_key_host, tail_key_host);
  const Tensor combined_value = concatenate_token_axis(history_value_host, tail_value_host);
  Tensor expected_output = Tensor::empty({heads, tail_tokens, value_dimension});
  Tensor expected_workspace =
      Tensor::empty({heads, tail_tokens, static_cast<Dimension>(history_tokens + tail_tokens)});
  cpu::scaled_dot_product_attention_out(query_host, combined_key, combined_value, history_count,
                                        expected_output, expected_workspace);

  const auto physical_dimension = static_cast<Dimension>(physical_blocks);
  const Tensor query = query_host.to(Device::cuda(0));
  const Tensor key_blocks =
      pack_logical_blocks(history_key_host, block_ids, physical_dimension, block_tokens, 10'000.0F)
          .to(Device::cuda(0));
  const Tensor value_blocks = pack_logical_blocks(history_value_host, block_ids, physical_dimension,
                                                  block_tokens, -10'000.0F)
                                  .to(Device::cuda(0));
  const Tensor tail_key = tail_key_host.to(Device::cuda(0));
  const Tensor tail_value = tail_value_host.to(Device::cuda(0));
  PagedBlockTable block_table(block_ids, physical_blocks, 0);
  Tensor output =
      Tensor::empty({heads, tail_tokens, value_dimension}, DType::kFloat32, Device::cuda(0));
  Tensor workspace = Tensor::empty(expected_workspace.shape(), DType::kFloat32, Device::cuda(0));
  helix::cuda::Stream stream;
  for (int launch = 0; launch < kWarmupLaunches; ++launch) {
    paged_scaled_dot_product_attention_with_tail_out(query, key_blocks, value_blocks, block_table,
                                                     history_count, tail_key, tail_value, output,
                                                     workspace, stream);
  }
  stream.synchronize();

  const float score_tolerance = 2.0e-5F + static_cast<float>(head_dimension) * 1.0e-6F;
  const float output_tolerance =
      score_tolerance + static_cast<float>(history_tokens + tail_tokens) * 2.0e-7F;
  if (!all_close(output, expected_output, output_tolerance) ||
      !all_close(workspace, expected_workspace, score_tolerance)) {
    state.SkipWithError("CUDA paged-tail attention failed its CPU-reference check");
    return;
  }

  helix::cuda::Event start;
  helix::cuda::Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    paged_scaled_dot_product_attention_with_tail_out(query, key_blocks, value_blocks, block_table,
                                                     history_count, tail_key, tail_value, output,
                                                     workspace, stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(helix::cuda::elapsed_time_ms(start, stop)) / 1000.0);
  }

  if (!all_close(output, expected_output, output_tolerance)) {
    state.SkipWithError("timed CUDA paged-tail attention diverged from its CPU reference");
    return;
  }
  const std::int64_t total_tokens = state.range(1) + state.range(2);
  const std::int64_t dot_flops =
      2 * state.range(0) * state.range(2) * total_tokens * (state.range(3) + state.range(4));
  state.counters["dot_FLOP/s"] = benchmark::Counter(static_cast<double>(dot_flops),
                                                    benchmark::Counter::kIsIterationInvariantRate);
  state.counters["workspace_bytes"] = static_cast<double>(workspace.nbytes());
  state.counters["block_table_bytes"] = static_cast<double>(block_table.size_bytes());
  state.counters["logical_kv_bytes"] =
      static_cast<double>(heads) * static_cast<double>(total_tokens) *
      static_cast<double>(head_dimension + value_dimension) * static_cast<double>(sizeof(float));
  state.counters["physical_history_kv_bytes"] =
      static_cast<double>(physical_blocks) * static_cast<double>(heads) *
      static_cast<double>(block_tokens) * static_cast<double>(head_dimension + value_dimension) *
      static_cast<double>(sizeof(float));
  state.SetItemsProcessed(state.iterations() * state.range(0) * state.range(2));
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

BENCHMARK(BM_CudaPagedAttentionWithTail)
    ->Args({8, 0, 1, 64, 64, 16})
    ->Args({8, 127, 1, 64, 64, 16})
    ->Args({8, 1023, 1, 64, 64, 16})
    ->Args({3, 26, 3, 37, 41, 8})
    ->UseManualTime();

} // namespace
} // namespace helix::attention::cuda
