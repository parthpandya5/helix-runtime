#include "helix/runtime/cpu_engine.hpp"

#include <benchmark/benchmark.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace helix::runtime {
namespace {

Tensor random_tensor(Shape shape, std::uint64_t seed, bool normalization = false) {
  return Tensor::random_uniform(std::move(shape), seed, normalization ? 0.8F : -0.08F,
                                normalization ? 1.2F : 0.08F);
}

model::LlamaWeights benchmark_weights(const model::LlamaConfig& config) {
  const auto vocabulary = static_cast<Dimension>(config.vocabulary_size);
  const auto hidden = static_cast<Dimension>(config.hidden_size);
  const auto intermediate = static_cast<Dimension>(config.intermediate_size);
  std::vector<model::LlamaLayerWeights> layers;
  layers.reserve(config.layer_count);
  for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
    const std::uint64_t seed = 40'000 + layer * 17;
    layers.emplace_back(
        random_tensor({hidden}, seed, true), random_tensor({hidden, hidden}, seed + 1),
        random_tensor({hidden, hidden}, seed + 2), random_tensor({hidden, hidden}, seed + 3),
        random_tensor({hidden, hidden}, seed + 4), random_tensor({hidden}, seed + 5, true),
        random_tensor({hidden, intermediate}, seed + 6),
        random_tensor({hidden, intermediate}, seed + 7),
        random_tensor({intermediate, hidden}, seed + 8));
  }
  return model::LlamaWeights(random_tensor({vocabulary, hidden}, 39'001), std::move(layers),
                             random_tensor({hidden}, 39'002, true),
                             random_tensor({hidden, vocabulary}, 39'003));
}

struct RuntimeTraceConfig {
  model::LlamaConfig model;
  CpuRuntimeConfig runtime;
  std::size_t request_count = 0;
  std::size_t prompt_tokens = 0;
  std::size_t long_output_tokens = 0;
};

std::size_t blocks_for_tokens(std::size_t tokens, std::size_t block_tokens) {
  return tokens / block_tokens + static_cast<std::size_t>(tokens % block_tokens != 0);
}

RuntimeTraceConfig trace_config(const benchmark::State& state) {
  constexpr std::size_t kVocabularySize = 128;
  constexpr std::size_t kBlockTokens = 4;
  const auto hidden = static_cast<std::size_t>(state.range(0));
  const auto intermediate = static_cast<std::size_t>(state.range(1));
  const auto layers = static_cast<std::size_t>(state.range(2));
  const auto heads = static_cast<std::size_t>(state.range(3));
  const auto request_count = static_cast<std::size_t>(state.range(4));
  const auto prompt_tokens = static_cast<std::size_t>(state.range(5));
  const auto output_tokens = static_cast<std::size_t>(state.range(6));
  const auto max_running = static_cast<std::size_t>(state.range(7));
  const std::size_t maximum_cached_tokens = prompt_tokens + output_tokens - 1;
  const std::size_t physical_blocks =
      max_running * blocks_for_tokens(maximum_cached_tokens, kBlockTokens);
  return RuntimeTraceConfig{model::LlamaConfig{kVocabularySize, hidden, intermediate, layers, heads,
                                               prompt_tokens + output_tokens, 1.0e-5F, 10'000.0F},
                            CpuRuntimeConfig{scheduler::SchedulerConfig{max_running * prompt_tokens,
                                                                        max_running, kBlockTokens},
                                             physical_blocks},
                            request_count, prompt_tokens, output_tokens};
}

std::vector<scheduler::TokenId> make_prompt(const RuntimeTraceConfig& config,
                                            std::size_t request_index) {
  std::vector<scheduler::TokenId> prompt;
  prompt.reserve(config.prompt_tokens);
  for (std::size_t token = 0; token < config.prompt_tokens; ++token) {
    prompt.emplace_back(static_cast<std::uint32_t>((request_index * 29 + token * 13 + 5) %
                                                   config.model.vocabulary_size));
  }
  return prompt;
}

std::size_t request_output_tokens(const RuntimeTraceConfig& config, std::size_t request_index) {
  const std::size_t max_running = config.runtime.scheduler.max_running_requests;
  return request_index % max_running == 0 ? config.long_output_tokens
                                          : (config.long_output_tokens + 3) / 4;
}

std::unique_ptr<CpuRuntimeEngine> make_engine(const RuntimeTraceConfig& config) {
  auto engine = std::make_unique<CpuRuntimeEngine>(
      model::CpuLlamaModel(config.model, benchmark_weights(config.model)), config.runtime);
  constexpr scheduler::RequestTimePoint kCreated{};
  for (std::size_t request = 0; request < config.request_count; ++request) {
    engine->submit(scheduler::Request(scheduler::RequestId{request + 1},
                                      make_prompt(config, request),
                                      request_output_tokens(config, request), kCreated));
  }
  return engine;
}

std::uint64_t execute_trace(CpuRuntimeEngine& engine) {
  constexpr std::size_t kIterationLimit = 1'000'000;
  scheduler::RequestTimePoint now{};
  std::uint64_t executed_iterations = 0;
  for (std::size_t iteration = 0; engine.has_live_requests(); ++iteration) {
    if (iteration == kIterationLimit) {
      throw std::runtime_error("CPU runtime benchmark trace exceeded its progress limit");
    }
    const CpuRuntimeIterationResult result = engine.run_iteration(now);
    if (!result.iteration_id.has_value()) {
      throw std::runtime_error("CPU runtime benchmark trace made no progress");
    }
    now += std::chrono::microseconds(1);
    ++executed_iterations;
  }
  return executed_iterations;
}

std::size_t tokens_per_trace(const RuntimeTraceConfig& config) {
  std::size_t tokens = 0;
  for (std::size_t request = 0; request < config.request_count; ++request) {
    tokens += config.prompt_tokens + request_output_tokens(config, request);
  }
  return tokens;
}

void run_runtime_trace_benchmark(benchmark::State& state, CpuPrefillMode prefill_mode,
                                 model::PagedDecodeMode paged_decode_mode) {
  RuntimeTraceConfig config = trace_config(state);
  config.runtime.prefill_mode = prefill_mode;
  config.runtime.paged_decode_mode = paged_decode_mode;
  bool warmed_up = false;
  std::uint64_t total_runtime_iterations = 0;
  std::size_t peak_kv_bytes = 0;
  for ([[maybe_unused]] const auto iteration : state) {
    state.PauseTiming();
    if (!warmed_up) {
      std::unique_ptr<CpuRuntimeEngine> warmup = make_engine(config);
      std::uint64_t warmup_iterations = execute_trace(*warmup);
      benchmark::DoNotOptimize(warmup_iterations);
      warmed_up = true;
    }
    std::unique_ptr<CpuRuntimeEngine> engine = make_engine(config);
    state.ResumeTiming();

    const std::uint64_t trace_iterations = execute_trace(*engine);

    state.PauseTiming();
    const CpuRuntimeStats stats = engine->stats();
    if (stats.scheduler.finished_count != config.request_count ||
        stats.scheduler.failed_count != 0 || stats.active_session_count != 0 ||
        stats.kv_cache.used_blocks != 0) {
      state.SkipWithError("CPU runtime trace ended with inconsistent ownership or request state");
    }
    total_runtime_iterations += trace_iterations;
    peak_kv_bytes = stats.kv_cache.peak_bytes_used;
    engine.reset();
    state.ResumeTiming();
  }

  const std::size_t token_count = tokens_per_trace(config);
  state.counters["runtime_iterations"] =
      static_cast<double>(total_runtime_iterations) / static_cast<double>(state.iterations());
  state.counters["peak_kv_bytes"] = static_cast<double>(peak_kv_bytes);
  const benchmark::IterationCount processed_requests =
      state.iterations() * static_cast<benchmark::IterationCount>(config.request_count);
  state.counters["requests_per_second"] =
      benchmark::Counter(static_cast<double>(processed_requests), benchmark::Counter::kIsRate);
  state.SetItemsProcessed(state.iterations() * static_cast<benchmark::IterationCount>(token_count));
}

void BM_CpuRuntimeContinuousTrace(benchmark::State& state, CpuPrefillMode prefill_mode) {
  run_runtime_trace_benchmark(state, prefill_mode, model::PagedDecodeMode::kCompactCopy);
}

void BM_CpuRuntimePagedDecodeTrace(benchmark::State& state,
                                   model::PagedDecodeMode paged_decode_mode) {
  run_runtime_trace_benchmark(state, CpuPrefillMode::kFullPrompt, paged_decode_mode);
}

BENCHMARK_CAPTURE(BM_CpuRuntimeContinuousTrace, SequentialTokenPrefill,
                  CpuPrefillMode::kSequentialDecode)
    ->Args({16, 32, 1, 4, 8, 4, 4, 4})
    ->Args({32, 64, 2, 4, 8, 8, 8, 4})
    ->Args({64, 128, 2, 4, 16, 8, 8, 8})
    ->ArgNames({"hidden", "intermediate", "layers", "heads", "requests", "prompt", "output",
                "active"});
BENCHMARK_CAPTURE(BM_CpuRuntimeContinuousTrace, FullPromptPrefill, CpuPrefillMode::kFullPrompt)
    ->Args({16, 32, 1, 4, 8, 4, 4, 4})
    ->Args({32, 64, 2, 4, 8, 8, 8, 4})
    ->Args({64, 128, 2, 4, 16, 8, 8, 8})
    ->ArgNames({"hidden", "intermediate", "layers", "heads", "requests", "prompt", "output",
                "active"});

BENCHMARK_CAPTURE(BM_CpuRuntimePagedDecodeTrace, CompactCopy, model::PagedDecodeMode::kCompactCopy)
    ->Args({16, 32, 1, 4, 8, 4, 4, 4})
    ->Args({32, 64, 2, 4, 8, 8, 8, 4})
    ->Args({64, 128, 2, 4, 16, 8, 8, 8})
    ->ArgNames({"hidden", "intermediate", "layers", "heads", "requests", "prompt", "output",
                "active"});
BENCHMARK_CAPTURE(BM_CpuRuntimePagedDecodeTrace, DirectBlockTable,
                  model::PagedDecodeMode::kDirectBlockTable)
    ->Args({16, 32, 1, 4, 8, 4, 4, 4})
    ->Args({32, 64, 2, 4, 8, 8, 8, 4})
    ->Args({64, 128, 2, 4, 16, 8, 8, 8})
    ->ArgNames({"hidden", "intermediate", "layers", "heads", "requests", "prompt", "output",
                "active"});

} // namespace
} // namespace helix::runtime
