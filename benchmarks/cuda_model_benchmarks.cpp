#include "helix/model/cuda_llama.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <array>
#include <benchmark/benchmark.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace helix::model {
namespace {

constexpr int kWarmupIterations = 10;
constexpr float kAbsoluteTolerance = 2.0e-4F;
constexpr float kRelativeTolerance = 2.0e-3F;

Tensor random_tensor(Shape shape, std::uint64_t seed, bool normalization = false) {
  return Tensor::random_uniform(std::move(shape), seed, normalization ? 0.8F : -0.08F,
                                normalization ? 1.2F : 0.08F);
}

LlamaWeights benchmark_weights(const LlamaConfig& config) {
  const Dimension vocabulary = static_cast<Dimension>(config.vocabulary_size);
  const Dimension hidden = static_cast<Dimension>(config.hidden_size);
  const Dimension intermediate = static_cast<Dimension>(config.intermediate_size);
  std::vector<LlamaLayerWeights> layers;
  layers.reserve(config.layer_count);
  for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
    const std::uint64_t seed = 1'000 + layer * 17;
    layers.emplace_back(
        random_tensor({hidden}, seed, true), random_tensor({hidden, hidden}, seed + 1),
        random_tensor({hidden, hidden}, seed + 2), random_tensor({hidden, hidden}, seed + 3),
        random_tensor({hidden, hidden}, seed + 4), random_tensor({hidden}, seed + 5, true),
        random_tensor({hidden, intermediate}, seed + 6),
        random_tensor({hidden, intermediate}, seed + 7),
        random_tensor({intermediate, hidden}, seed + 8));
  }
  return LlamaWeights(random_tensor({vocabulary, hidden}, 1), std::move(layers),
                      random_tensor({hidden}, 2, true), random_tensor({hidden, vocabulary}, 3));
}

LlamaConfig config_from_state(const benchmark::State& state) {
  return LlamaConfig{static_cast<std::size_t>(state.range(5)),
                     static_cast<std::size_t>(state.range(0)),
                     static_cast<std::size_t>(state.range(1)),
                     static_cast<std::size_t>(state.range(2)),
                     static_cast<std::size_t>(state.range(3)),
                     static_cast<std::size_t>(state.range(4)),
                     1.0e-5F,
                     10'000.0F};
}

std::vector<std::uint32_t> make_prompt(const LlamaConfig& config) {
  std::vector<std::uint32_t> result(config.max_sequence_length);
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<std::uint32_t>((index * 13 + 5) % config.vocabulary_size);
  }
  return result;
}

std::pair<Tensor, Tensor> make_rope_cache(const LlamaConfig& config) {
  const std::size_t tokens = config.max_sequence_length;
  const std::size_t head_dimension = config.head_dimension();
  const std::size_t pairs = head_dimension / 2;
  std::vector<float> cosine(tokens * pairs);
  std::vector<float> sine(tokens * pairs);
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t pair = 0; pair < pairs; ++pair) {
      const double exponent =
          -2.0 * static_cast<double>(pair) / static_cast<double>(head_dimension);
      const double angle =
          static_cast<double>(token) * std::pow(static_cast<double>(config.rope_theta), exponent);
      cosine[token * pairs + pair] = static_cast<float>(std::cos(angle));
      sine[token * pairs + pair] = static_cast<float>(std::sin(angle));
    }
  }
  const Shape shape{static_cast<Dimension>(tokens), static_cast<Dimension>(pairs)};
  return {Tensor::from_values(shape, cosine, Device::cuda(0)),
          Tensor::from_values(shape, sine, Device::cuda(0))};
}

std::pair<Tensor, Tensor> make_rope_row(const LlamaConfig& config, std::size_t position) {
  const std::size_t head_dimension = config.head_dimension();
  const std::size_t pairs = head_dimension / 2;
  std::vector<float> cosine(pairs);
  std::vector<float> sine(pairs);
  for (std::size_t pair = 0; pair < pairs; ++pair) {
    const double exponent = -2.0 * static_cast<double>(pair) / static_cast<double>(head_dimension);
    const double angle =
        static_cast<double>(position) * std::pow(static_cast<double>(config.rope_theta), exponent);
    cosine[pair] = static_cast<float>(std::cos(angle));
    sine[pair] = static_cast<float>(std::sin(angle));
  }
  const Shape shape{1, static_cast<Dimension>(pairs)};
  return {Tensor::from_values(shape, cosine, Device::cuda(0)),
          Tensor::from_values(shape, sine, Device::cuda(0))};
}

Tensor pack_logical_blocks(const Tensor& logical, std::span<const std::uint32_t> block_table,
                           Dimension physical_blocks, Dimension block_tokens, float padding_value) {
  const auto heads = static_cast<std::size_t>(logical.shape()[0]);
  const auto tokens = static_cast<std::size_t>(logical.shape()[1]);
  const auto width = static_cast<std::size_t>(logical.shape()[2]);
  const auto physical_count = static_cast<std::size_t>(physical_blocks);
  const auto tokens_per_block = static_cast<std::size_t>(block_tokens);
  std::vector<float> packed(physical_count * heads * tokens_per_block * width, padding_value);
  const float* source = logical.data_as<float>();
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
      {physical_blocks, logical.shape()[0], block_tokens, logical.shape()[2]}, packed,
      Device::cuda(0));
}

std::vector<std::uint32_t> reversed_block_table(std::size_t logical_blocks,
                                                std::size_t physical_blocks) {
  std::vector<std::uint32_t> result(logical_blocks);
  for (std::size_t logical = 0; logical < logical_blocks; ++logical) {
    result[logical] = static_cast<std::uint32_t>(physical_blocks - 1 - logical);
  }
  return result;
}

bool output_matches(benchmark::State& state, const Tensor& actual_device,
                    const Tensor& expected_host) {
  const Tensor actual = actual_device.to(Device::cpu());
  if (actual.shape() != expected_host.shape()) {
    state.SkipWithError("CUDA Llama output shape does not match the CPU oracle");
    return false;
  }
  for (std::size_t index = 0; index < actual.numel(); ++index) {
    const float actual_value = actual.data_as<float>()[index];
    const float expected_value = expected_host.data_as<float>()[index];
    const float allowed_error = kAbsoluteTolerance + kRelativeTolerance * std::abs(expected_value);
    if (!std::isfinite(actual_value) || std::abs(actual_value - expected_value) > allowed_error) {
      state.SkipWithError("CUDA Llama output does not match the CPU oracle");
      return false;
    }
  }
  return true;
}

bool decode_matches(benchmark::State& state, const CudaLlamaDecodeResult& actual,
                    const CpuLlamaDecodeResult& expected) {
  return output_matches(state, actual.token_keys, expected.token_keys) &&
         output_matches(state, actual.token_values, expected.token_values) &&
         output_matches(state, actual.logits, expected.logits);
}

std::size_t parameter_count(const LlamaConfig& config) {
  const std::size_t embedding_and_head = 2 * config.vocabulary_size * config.hidden_size;
  const std::size_t per_layer = 2 * config.hidden_size +
                                4 * config.hidden_size * config.hidden_size +
                                3 * config.hidden_size * config.intermediate_size;
  return embedding_and_head + config.hidden_size + config.layer_count * per_layer;
}

void BM_CudaLlamaSynchronousPrefill(benchmark::State& state) {
  if (!helix::cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  helix::cuda::set_device(0);
  const LlamaConfig config = config_from_state(state);
  const std::vector<std::uint32_t> prompt = make_prompt(config);
  CpuLlamaModel cpu_model(config, benchmark_weights(config));
  const Tensor expected = cpu_model.forward(prompt);
  const LlamaWeights source_weights = benchmark_weights(config);
  CudaLlamaModel model(config, source_weights);
  const Tensor token_ids =
      Tensor::from_values({static_cast<Dimension>(prompt.size())}, prompt, Device::cuda(0));
  const auto [cosine, sine] = make_rope_cache(config);
  helix::cuda::Stream stream;

  Tensor verification = model.forward_device(token_ids, cosine, sine, stream);
  if (!output_matches(state, verification, expected)) {
    return;
  }
  for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
    Tensor warmup = model.forward_device(token_ids, cosine, sine, stream);
    benchmark::DoNotOptimize(warmup.data());
  }

  for ([[maybe_unused]] auto iteration : state) {
    const auto start = std::chrono::steady_clock::now();
    Tensor logits = model.forward_device(token_ids, cosine, sine, stream);
    const auto stop = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(stop - start).count());
    benchmark::DoNotOptimize(logits.data());
  }

  verification = model.forward_device(token_ids, cosine, sine, stream);
  if (!output_matches(state, verification, expected)) {
    return;
  }
  const std::size_t parameters = parameter_count(config);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(prompt.size()));
  state.counters["parameters"] = static_cast<double>(parameters);
  state.counters["parameter_bytes"] = static_cast<double>(parameters * sizeof(float));
  state.counters["synchronizations"] = static_cast<double>(config.layer_count + 1);
  state.counters["tokens"] = static_cast<double>(prompt.size());
  state.counters["tokens/s"] = benchmark::Counter(static_cast<double>(prompt.size()),
                                                  benchmark::Counter::kIsIterationInvariantRate);
}

void BM_CudaLlamaSynchronousPagedDecode(benchmark::State& state) {
  if (!helix::cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  helix::cuda::set_device(0);
  const std::size_t history_tokens = static_cast<std::size_t>(state.range(4));
  const std::size_t block_tokens = static_cast<std::size_t>(state.range(6));
  const LlamaConfig config{static_cast<std::size_t>(state.range(5)),
                           static_cast<std::size_t>(state.range(0)),
                           static_cast<std::size_t>(state.range(1)),
                           static_cast<std::size_t>(state.range(2)),
                           static_cast<std::size_t>(state.range(3)),
                           history_tokens + 1,
                           1.0e-5F,
                           10'000.0F};
  std::vector<std::uint32_t> prefix(history_tokens);
  for (std::size_t index = 0; index < history_tokens; ++index) {
    prefix[index] = static_cast<std::uint32_t>((index * 13 + 5) % config.vocabulary_size);
  }
  const std::uint32_t token_id =
      static_cast<std::uint32_t>((history_tokens * 13 + 5) % config.vocabulary_size);

  CpuLlamaModel cpu_model(config, benchmark_weights(config));
  std::vector<Tensor> logical_keys;
  std::vector<Tensor> logical_values;
  logical_keys.reserve(config.layer_count);
  logical_values.reserve(config.layer_count);
  if (history_tokens == 0) {
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
      logical_keys.push_back(Tensor::zeros({static_cast<Dimension>(config.attention_head_count), 0,
                                            static_cast<Dimension>(config.head_dimension())}));
      logical_values.push_back(Tensor::zeros(logical_keys.back().shape()));
    }
  } else {
    CpuLlamaPrefillResult prefill = cpu_model.prefill_with_kv(prefix);
    logical_keys = std::move(prefill.layer_keys);
    logical_values = std::move(prefill.layer_values);
  }

  const std::size_t logical_blocks =
      history_tokens == 0 ? 0 : 1 + (history_tokens - 1) / block_tokens;
  const std::size_t physical_blocks = logical_blocks + 2;
  const std::vector<std::uint32_t> table = reversed_block_table(logical_blocks, physical_blocks);
  const auto physical_dimension = static_cast<Dimension>(physical_blocks);
  const auto block_dimension = static_cast<Dimension>(block_tokens);
  std::vector<Tensor> key_blocks;
  std::vector<Tensor> value_blocks;
  key_blocks.reserve(config.layer_count);
  value_blocks.reserve(config.layer_count);
  for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
    key_blocks.push_back(pack_logical_blocks(logical_keys[layer], table, physical_dimension,
                                             block_dimension,
                                             std::numeric_limits<float>::quiet_NaN()));
    value_blocks.push_back(pack_logical_blocks(logical_values[layer], table, physical_dimension,
                                               block_dimension,
                                               std::numeric_limits<float>::quiet_NaN()));
  }
  attention::cuda::PagedBlockTable block_table(table, physical_blocks, 0);

  kv_cache::PagedKvCache cpu_cache(
      kv_cache::PagedKvCacheConfig{config.layer_count, config.attention_head_count,
                                   config.head_dimension(), block_tokens, physical_blocks});
  const kv_cache::PagedSequenceHandle sequence = cpu_cache.create_sequence();
  if (history_tokens != 0) {
    cpu_cache.append_layer_major(sequence, logical_keys, logical_values);
  }
  const CpuLlamaDecodeResult expected =
      cpu_model.decode_one(token_id, history_tokens, cpu_cache, sequence);

  const LlamaWeights source_weights = benchmark_weights(config);
  CudaLlamaModel model(config, source_weights);
  const std::array<std::uint32_t, 1> token_values{token_id};
  const Tensor device_id = Tensor::from_values({1}, token_values, Device::cuda(0));
  const auto [cosine, sine] = make_rope_row(config, history_tokens);
  helix::cuda::Stream stream;

  CudaLlamaDecodeResult verification = model.decode_one_paged_device(
      device_id, cosine, sine, key_blocks, value_blocks, block_table, history_tokens, stream);
  if (!decode_matches(state, verification, expected)) {
    return;
  }
  for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
    CudaLlamaDecodeResult warmup = model.decode_one_paged_device(
        device_id, cosine, sine, key_blocks, value_blocks, block_table, history_tokens, stream);
    benchmark::DoNotOptimize(warmup.logits.data());
  }

  for ([[maybe_unused]] auto iteration : state) {
    const auto start = std::chrono::steady_clock::now();
    CudaLlamaDecodeResult result = model.decode_one_paged_device(
        device_id, cosine, sine, key_blocks, value_blocks, block_table, history_tokens, stream);
    const auto stop = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(stop - start).count());
    benchmark::DoNotOptimize(result.logits.data());
  }

  verification = model.decode_one_paged_device(device_id, cosine, sine, key_blocks, value_blocks,
                                               block_table, history_tokens, stream);
  if (!decode_matches(state, verification, expected)) {
    return;
  }
  const std::size_t parameters = parameter_count(config);
  const std::size_t page_bytes = physical_blocks * config.layer_count *
                                 config.attention_head_count * block_tokens *
                                 config.head_dimension() * 2 * sizeof(float);
  state.SetItemsProcessed(state.iterations());
  state.counters["parameters"] = static_cast<double>(parameters);
  state.counters["parameter_bytes"] = static_cast<double>(parameters * sizeof(float));
  state.counters["history_tokens"] = static_cast<double>(history_tokens);
  state.counters["physical_kv_bytes"] = static_cast<double>(page_bytes);
  state.counters["block_table_bytes"] = static_cast<double>(block_table.size_bytes());
  state.counters["score_workspace_bytes_per_layer"] =
      static_cast<double>(config.attention_head_count * (history_tokens + 1) * sizeof(float));
  state.counters["synchronizations"] = static_cast<double>(config.layer_count + 1);
  state.counters["kv_result_copies"] = static_cast<double>(2 * config.layer_count);
  state.counters["tokens/s"] =
      benchmark::Counter(1.0, benchmark::Counter::kIsIterationInvariantRate);
}

void model_arguments(benchmark::internal::Benchmark* benchmark) {
  benchmark->Args({32, 64, 2, 4, 7, 128});
  benchmark->Args({64, 128, 2, 4, 31, 256});
  benchmark->Args({128, 256, 4, 8, 63, 512});
}

BENCHMARK(BM_CudaLlamaSynchronousPrefill)
    ->Apply(model_arguments)
    ->ArgNames({"hidden", "intermediate", "layers", "heads", "tokens", "vocabulary"})
    ->UseManualTime();

BENCHMARK(BM_CudaLlamaSynchronousPagedDecode)
    ->Args({32, 64, 2, 4, 0, 128, 8})
    ->Args({64, 128, 2, 4, 31, 256, 8})
    ->Args({128, 256, 4, 8, 127, 512, 16})
    ->ArgNames({"hidden", "intermediate", "layers", "heads", "history", "vocabulary",
                "block_tokens"})
    ->UseManualTime();

} // namespace
} // namespace helix::model
