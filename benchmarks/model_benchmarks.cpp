#include "helix/model/format.hpp"
#include "helix/model/generation.hpp"
#include "helix/model/llama.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace helix::model {
namespace {

Tensor random_tensor(Shape shape, std::uint64_t seed, bool normalization = false) {
  return Tensor::random_uniform(std::move(shape), seed, normalization ? 0.8F : -0.08F,
                                normalization ? 1.2F : 0.08F);
}

LlamaWeights benchmark_weights(const LlamaConfig& config) {
  const auto vocabulary = static_cast<Dimension>(config.vocabulary_size);
  const auto hidden = static_cast<Dimension>(config.hidden_size);
  const auto intermediate = static_cast<Dimension>(config.intermediate_size);
  std::vector<LlamaLayerWeights> layers;
  layers.reserve(config.layer_count);
  for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
    const std::uint64_t seed = 100 + layer * 17;
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

std::size_t parameter_count(const LlamaConfig& config) {
  const std::size_t embedding_and_head = 2 * config.vocabulary_size * config.hidden_size;
  const std::size_t per_layer = 2 * config.hidden_size +
                                4 * config.hidden_size * config.hidden_size +
                                3 * config.hidden_size * config.intermediate_size;
  return embedding_and_head + config.hidden_size + config.layer_count * per_layer;
}

void set_model_counters(benchmark::State& state, const LlamaConfig& config,
                        std::size_t model_file_bytes = 0) {
  const std::size_t parameters = parameter_count(config);
  state.counters["parameters"] = static_cast<double>(parameters);
  state.counters["parameter_bytes"] = static_cast<double>(parameters * sizeof(float));
  state.counters["sequence_tokens"] = static_cast<double>(config.max_sequence_length);
  if (model_file_bytes != 0) {
    state.counters["model_file_bytes"] = static_cast<double>(model_file_bytes);
  }
}

void BM_CpuLlamaFullPrefill(benchmark::State& state) {
  const LlamaConfig config = config_from_state(state);
  std::optional<CpuLlamaModel> model;
  std::vector<std::uint32_t> token_ids;

  for ([[maybe_unused]] const auto iteration : state) {
    if (!model.has_value()) {
      state.PauseTiming();
      model.emplace(config, benchmark_weights(config));
      token_ids.resize(config.max_sequence_length);
      for (std::size_t index = 0; index < token_ids.size(); ++index) {
        token_ids[index] = static_cast<std::uint32_t>((index * 13 + 5) % config.vocabulary_size);
      }
      const Tensor warmup = model->forward(token_ids);
      benchmark::DoNotOptimize(warmup.data());
      state.ResumeTiming();
    }
    const Tensor logits = model->forward(token_ids);
    benchmark::DoNotOptimize(logits.data());
    benchmark::ClobberMemory();
  }

  set_model_counters(state, config);
  state.counters["tokens_per_second"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * static_cast<double>(config.max_sequence_length),
      benchmark::Counter::kIsRate);
}

void BM_LlamaModelLoadFromMemory(benchmark::State& state) {
  const LlamaConfig config = config_from_state(state);
  std::optional<std::string> serialized;

  for ([[maybe_unused]] const auto iteration : state) {
    if (!serialized.has_value()) {
      state.PauseTiming();
      const LlamaWeights weights = benchmark_weights(config);
      std::ostringstream output(std::ios::out | std::ios::binary);
      write_llama_model_file(output, config, weights);
      serialized = output.str();
      std::istringstream warmup_input(*serialized, std::ios::in | std::ios::binary);
      const LoadedLlamaModel warmup = read_llama_model_file(warmup_input);
      benchmark::DoNotOptimize(warmup.weights.token_embedding().data());
      state.ResumeTiming();
    }
    std::istringstream input(*serialized, std::ios::in | std::ios::binary);
    const LoadedLlamaModel loaded = read_llama_model_file(input);
    benchmark::DoNotOptimize(loaded.weights.token_embedding().data());
  }

  set_model_counters(state, config, serialized->size());
}

std::vector<std::uint32_t> make_prompt(std::size_t token_count, std::size_t vocabulary_size) {
  std::vector<std::uint32_t> token_ids(token_count);
  for (std::size_t index = 0; index < token_ids.size(); ++index) {
    token_ids[index] = static_cast<std::uint32_t>((index * 13 + 5) % vocabulary_size);
  }
  return token_ids;
}

void BM_CpuLlamaNextPosition(benchmark::State& state, bool cached_decode) {
  LlamaConfig config = config_from_state(state);
  const std::size_t prompt_tokens = config.max_sequence_length;
  ++config.max_sequence_length;
  const std::vector<std::uint32_t> prompt = make_prompt(prompt_tokens, config.vocabulary_size);
  const std::uint32_t decode_token =
      static_cast<std::uint32_t>((prompt_tokens * 17 + 3) % config.vocabulary_size);
  std::optional<CpuLlamaModel> full_model;
  std::optional<CpuLlamaGenerator> generator;
  std::vector<std::uint32_t> full_context = prompt;
  full_context.push_back(decode_token);

  for ([[maybe_unused]] const auto iteration : state) {
    if (cached_decode) {
      state.PauseTiming();
      if (!generator.has_value()) {
        CpuLlamaGenerator warmup(CpuLlamaModel(config, benchmark_weights(config)));
        static_cast<void>(warmup.prefill(prompt));
        const Tensor warmup_logits = warmup.decode(decode_token);
        benchmark::DoNotOptimize(warmup_logits.data());
      }
      generator.reset();
      generator.emplace(CpuLlamaModel(config, benchmark_weights(config)));
      static_cast<void>(generator->prefill(prompt));
      state.ResumeTiming();
      const Tensor logits = generator->decode(decode_token);
      benchmark::DoNotOptimize(logits.data());
    } else {
      if (!full_model.has_value()) {
        state.PauseTiming();
        full_model.emplace(config, benchmark_weights(config));
        const Tensor warmup_logits = full_model->forward(full_context);
        benchmark::DoNotOptimize(warmup_logits.data());
        state.ResumeTiming();
      }
      const Tensor logits = full_model->forward(full_context);
      benchmark::DoNotOptimize(logits.data());
    }
    benchmark::ClobberMemory();
  }

  set_model_counters(state, config);
  state.counters["prompt_tokens"] = static_cast<double>(prompt_tokens);
  state.counters["positions_per_second"] =
      benchmark::Counter(static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

void model_arguments(benchmark::internal::Benchmark* benchmark) {
  benchmark->Args({32, 64, 2, 4, 7, 128});
  benchmark->Args({64, 128, 2, 4, 31, 256});
  benchmark->Args({128, 256, 4, 8, 63, 512});
}

BENCHMARK(BM_CpuLlamaFullPrefill)
    ->Apply(model_arguments)
    ->ArgNames({"hidden", "intermediate", "layers", "heads", "tokens", "vocabulary"});
BENCHMARK(BM_LlamaModelLoadFromMemory)
    ->Apply(model_arguments)
    ->ArgNames({"hidden", "intermediate", "layers", "heads", "tokens", "vocabulary"});
BENCHMARK_CAPTURE(BM_CpuLlamaNextPosition, CachedKvDecode, true)
    ->Apply(model_arguments)
    ->ArgNames({"hidden", "intermediate", "layers", "heads", "prompt", "vocabulary"});
BENCHMARK_CAPTURE(BM_CpuLlamaNextPosition, FullPrefixRecompute, false)
    ->Apply(model_arguments)
    ->ArgNames({"hidden", "intermediate", "layers", "heads", "prompt", "vocabulary"});

} // namespace
} // namespace helix::model
