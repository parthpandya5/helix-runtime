#include "helix/model/cuda_llama.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

  helix::cuda::Event start;
  helix::cuda::Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    Tensor logits = model.forward_device(token_ids, cosine, sine, stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(helix::cuda::elapsed_time_ms(start, stop)) / 1000.0);
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

void model_arguments(benchmark::internal::Benchmark* benchmark) {
  benchmark->Args({32, 64, 2, 4, 7, 128});
  benchmark->Args({64, 128, 2, 4, 31, 256});
  benchmark->Args({128, 256, 4, 8, 63, 512});
}

BENCHMARK(BM_CudaLlamaSynchronousPrefill)
    ->Apply(model_arguments)
    ->ArgNames({"hidden", "intermediate", "layers", "heads", "tokens", "vocabulary"})
    ->UseManualTime();

} // namespace
} // namespace helix::model
