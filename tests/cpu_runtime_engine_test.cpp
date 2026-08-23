#include "helix/core/error.hpp"
#include "helix/model/generation.hpp"
#include "helix/runtime/cpu_engine.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace helix::runtime {
namespace {

using namespace std::chrono_literals;

constexpr scheduler::RequestTimePoint kStart{};

model::LlamaConfig runtime_model_config(std::size_t maximum_length = 10) {
  return model::LlamaConfig{11, 4, 7, 2, 2, maximum_length, 1.0e-5F, 10'000.0F};
}

Tensor make_tensor(Shape shape, std::size_t seed, bool normalization = false) {
  Tensor tensor = Tensor::empty(std::move(shape));
  float* values = tensor.data_as<float>();
  for (std::size_t index = 0; index < tensor.numel(); ++index) {
    values[index] =
        normalization
            ? 0.8F + static_cast<float>((index + seed) % 9) * 0.025F
            : static_cast<float>(static_cast<int>((index * 11 + seed * 7) % 31) - 15) * 0.0175F;
  }
  return tensor;
}

model::LlamaWeights make_weights(const model::LlamaConfig& config,
                                 Tensor* lm_head_alias = nullptr) {
  const auto vocabulary = static_cast<Dimension>(config.vocabulary_size);
  const auto hidden = static_cast<Dimension>(config.hidden_size);
  const auto intermediate = static_cast<Dimension>(config.intermediate_size);
  std::vector<model::LlamaLayerWeights> layers;
  layers.reserve(config.layer_count);
  for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
    const std::size_t seed = 20 + layer * 13;
    layers.emplace_back(
        make_tensor({hidden}, seed, true), make_tensor({hidden, hidden}, seed + 1),
        make_tensor({hidden, hidden}, seed + 2), make_tensor({hidden, hidden}, seed + 3),
        make_tensor({hidden, hidden}, seed + 4), make_tensor({hidden}, seed + 5, true),
        make_tensor({hidden, intermediate}, seed + 6),
        make_tensor({hidden, intermediate}, seed + 7),
        make_tensor({intermediate, hidden}, seed + 8));
  }
  Tensor lm_head = make_tensor({hidden, vocabulary}, 102);
  if (lm_head_alias != nullptr) {
    *lm_head_alias = lm_head;
  }
  return model::LlamaWeights(make_tensor({vocabulary, hidden}, 1), std::move(layers),
                             make_tensor({hidden}, 101, true), std::move(lm_head));
}

model::CpuLlamaModel make_model(const model::LlamaConfig& config, Tensor* lm_head_alias = nullptr) {
  return model::CpuLlamaModel(config, make_weights(config, lm_head_alias));
}

scheduler::Request make_request(std::uint64_t id, std::vector<std::uint32_t> prompt,
                                std::size_t max_new_tokens,
                                scheduler::RequestTimePoint created_at = kStart) {
  std::vector<scheduler::TokenId> tokens;
  tokens.reserve(prompt.size());
  for (const std::uint32_t token : prompt) {
    tokens.emplace_back(token);
  }
  return scheduler::Request(scheduler::RequestId{id}, std::move(tokens), max_new_tokens,
                            created_at);
}

std::vector<std::uint32_t> full_recompute_greedy(model::CpuLlamaModel& model,
                                                 std::vector<std::uint32_t> context,
                                                 std::size_t count) {
  std::vector<std::uint32_t> generated;
  generated.reserve(count);
  for (std::size_t step = 0; step < count; ++step) {
    const std::uint32_t token = model::select_greedy_token(model.forward(context));
    generated.push_back(token);
    if (step + 1 < count) {
      context.push_back(token);
    }
  }
  return generated;
}

std::vector<std::uint32_t> generated_values(const scheduler::Request& request) {
  std::vector<std::uint32_t> result;
  result.reserve(request.generated_tokens().size());
  for (const scheduler::GeneratedToken& token : request.generated_tokens()) {
    result.push_back(token.id.value());
  }
  return result;
}

void run_to_completion(CpuRuntimeEngine& engine, scheduler::RequestTimePoint start_at,
                       std::size_t maximum_iterations = 32) {
  for (std::size_t iteration = 0; iteration < maximum_iterations && engine.has_live_requests();
       ++iteration) {
    const CpuRuntimeIterationResult result =
        engine.run_iteration(start_at + std::chrono::milliseconds(iteration));
    ASSERT_TRUE(result.iteration_id.has_value()) << "runtime made no progress at " << iteration;
  }
  ASSERT_FALSE(engine.has_live_requests());
}

TEST(CpuRuntimeEngineTest, ExecutesChunkedPagedPrefillAndMatchesGreedyRecomputation) {
  const model::LlamaConfig model_config = runtime_model_config();
  model::CpuLlamaModel reference = make_model(model_config);
  const std::vector<std::uint32_t> prompt{2, 5, 1};
  const std::vector<std::uint32_t> expected = full_recompute_greedy(reference, prompt, 3);
  CpuRuntimeEngine engine(make_model(model_config),
                          CpuRuntimeConfig{scheduler::SchedulerConfig{2, 1, 2}, 8});
  engine.submit(make_request(1, prompt, 3));

  const CpuRuntimeIterationResult first = engine.run_iteration(kStart);
  ASSERT_TRUE(first.iteration_id.has_value());
  ASSERT_EQ(first.work.size(), 1U);
  EXPECT_EQ(first.work[0].kind, scheduler::WorkKind::kPrefill);
  EXPECT_EQ(first.work[0].token_count, 2U);
  EXPECT_EQ(engine.stats().kv_cache.active_tokens, 2U);

  run_to_completion(engine, kStart + 1ms);
  const scheduler::Request& completed = engine.request(scheduler::RequestId{1});
  EXPECT_EQ(completed.state(), scheduler::RequestState::kFinished);
  EXPECT_EQ(completed.finish_reason(), scheduler::FinishReason::kLength);
  EXPECT_EQ(generated_values(completed), expected);

  const CpuRuntimeStats stats = engine.stats();
  EXPECT_EQ(stats.executed_iteration_count, 5U);
  EXPECT_EQ(stats.failed_work_item_count, 0U);
  EXPECT_EQ(stats.active_session_count, 0U);
  EXPECT_EQ(stats.kv_cache.active_sequence_count, 0U);
  EXPECT_EQ(stats.kv_cache.active_tokens, 0U);
  EXPECT_EQ(stats.kv_cache.free_blocks, stats.kv_cache.total_blocks);
  EXPECT_EQ(stats.kv_cache.append_count, 5U);
  EXPECT_EQ(stats.full_prompt_prefill_count, 0U);
  EXPECT_EQ(stats.sequential_prefill_token_count, 3U);
}

TEST(CpuRuntimeEngineTest, FullPromptPrefillMatchesReferenceAndUsesBulkPagedAppend) {
  const model::LlamaConfig model_config = runtime_model_config(8);
  const std::vector<std::uint32_t> prompt{2, 5, 1, 7, 3};
  model::CpuLlamaModel reference = make_model(model_config);
  const std::vector<std::uint32_t> expected = full_recompute_greedy(reference, prompt, 1);
  CpuRuntimeEngine engine(make_model(model_config),
                          CpuRuntimeConfig{scheduler::SchedulerConfig{8, 1, 2}, 3});
  engine.submit(make_request(2, prompt, 1));

  const CpuRuntimeIterationResult prefill = engine.run_iteration(kStart);
  ASSERT_EQ(prefill.work.size(), 1U);
  EXPECT_EQ(prefill.work[0].token_count, prompt.size());
  CpuRuntimeStats stats = engine.stats();
  EXPECT_EQ(stats.full_prompt_prefill_count, 1U);
  EXPECT_EQ(stats.sequential_prefill_token_count, 0U);
  EXPECT_EQ(stats.kv_cache.append_count, prompt.size());
  EXPECT_EQ(stats.kv_cache.used_blocks, 3U);

  const CpuRuntimeIterationResult decode = engine.run_iteration(kStart + 1ms);
  ASSERT_EQ(decode.retired_requests.size(), 1U);
  EXPECT_EQ(generated_values(engine.request(scheduler::RequestId{2})), expected);
  stats = engine.stats();
  EXPECT_EQ(stats.kv_cache.used_blocks, 0U);
  EXPECT_EQ(stats.active_session_count, 0U);
}

TEST(CpuRuntimeEngineTest, DirectAndCompactedPagedDecodeModesProduceIdenticalRequests) {
  const model::LlamaConfig model_config = runtime_model_config();
  const std::vector<std::uint32_t> prompt{2, 5, 1};
  CpuRuntimeConfig direct_config{scheduler::SchedulerConfig{8, 2, 2}, 8};
  direct_config.paged_decode_mode = model::PagedDecodeMode::kDirectBlockTable;
  CpuRuntimeConfig compact_config = direct_config;
  compact_config.paged_decode_mode = model::PagedDecodeMode::kCompactCopy;
  CpuRuntimeEngine direct(make_model(model_config), direct_config);
  CpuRuntimeEngine compact(make_model(model_config), compact_config);
  direct.submit(make_request(3, prompt, 4));
  compact.submit(make_request(3, prompt, 4));

  run_to_completion(direct, kStart);
  run_to_completion(compact, kStart);
  const scheduler::Request& direct_request = direct.request(scheduler::RequestId{3});
  const scheduler::Request& compact_request = compact.request(scheduler::RequestId{3});
  EXPECT_EQ(generated_values(direct_request), generated_values(compact_request));
  EXPECT_EQ(direct_request.finish_reason(), compact_request.finish_reason());
  EXPECT_EQ(direct.stats().executed_iteration_count, compact.stats().executed_iteration_count);
  EXPECT_EQ(direct.stats().kv_cache.used_blocks, 0U);
  EXPECT_EQ(compact.stats().kv_cache.used_blocks, 0U);
}

TEST(CpuRuntimeEngineTest, RefillsFinishedSlotWithRealModelWork) {
  const model::LlamaConfig model_config = runtime_model_config();
  CpuRuntimeEngine engine(make_model(model_config),
                          CpuRuntimeConfig{scheduler::SchedulerConfig{4, 2, 2}, 8});
  engine.submit(make_request(10, {1}, 1));
  engine.submit(make_request(11, {2}, 3));
  engine.submit(make_request(12, {3}, 1));

  const CpuRuntimeIterationResult prefill = engine.run_iteration(kStart);
  ASSERT_EQ(prefill.work.size(), 2U);
  EXPECT_EQ(prefill.work[0].request_id, scheduler::RequestId{10});
  EXPECT_EQ(prefill.work[1].request_id, scheduler::RequestId{11});

  const CpuRuntimeIterationResult decode = engine.run_iteration(kStart + 1ms);
  ASSERT_EQ(decode.work.size(), 2U);
  ASSERT_EQ(decode.retired_requests.size(), 1U);
  EXPECT_EQ(decode.retired_requests[0].request_id, scheduler::RequestId{10});

  const CpuRuntimeIterationResult refill = engine.run_iteration(kStart + 2ms);
  ASSERT_EQ(refill.work.size(), 2U);
  EXPECT_EQ(refill.work[0].request_id, scheduler::RequestId{11});
  EXPECT_EQ(refill.work[0].kind, scheduler::WorkKind::kDecode);
  EXPECT_EQ(refill.work[1].request_id, scheduler::RequestId{12});
  EXPECT_EQ(refill.work[1].kind, scheduler::WorkKind::kPrefill);

  run_to_completion(engine, kStart + 3ms);
  EXPECT_EQ(engine.request(scheduler::RequestId{10}).state(), scheduler::RequestState::kFinished);
  EXPECT_EQ(engine.request(scheduler::RequestId{11}).state(), scheduler::RequestState::kFinished);
  EXPECT_EQ(engine.request(scheduler::RequestId{12}).state(), scheduler::RequestState::kFinished);
  EXPECT_EQ(engine.stats().kv_cache.used_blocks, 0U);
}

TEST(CpuRuntimeEngineTest, PhysicalKvPressureBlocksAdmissionUntilReclamation) {
  const model::LlamaConfig model_config = runtime_model_config(4);
  CpuRuntimeEngine engine(make_model(model_config),
                          CpuRuntimeConfig{scheduler::SchedulerConfig{4, 2, 2}, 1});
  engine.submit(make_request(20, {1, 2}, 1));
  engine.submit(make_request(21, {2, 3}, 1));

  const CpuRuntimeIterationResult first = engine.run_iteration(kStart);
  ASSERT_EQ(first.work.size(), 1U);
  EXPECT_EQ(first.work[0].request_id, scheduler::RequestId{20});
  EXPECT_TRUE(first.kv_capacity_limited);
  EXPECT_EQ(first.free_kv_blocks_after, 0U);

  const CpuRuntimeIterationResult retire = engine.run_iteration(kStart + 1ms);
  ASSERT_EQ(retire.retired_requests.size(), 1U);
  EXPECT_EQ(retire.retired_requests[0].request_id, scheduler::RequestId{20});
  EXPECT_EQ(retire.work[0].new_kv_blocks, 0U);
  EXPECT_EQ(retire.free_kv_blocks_after, 1U);

  const CpuRuntimeIterationResult replacement = engine.run_iteration(kStart + 2ms);
  ASSERT_EQ(replacement.work.size(), 1U);
  EXPECT_EQ(replacement.work[0].request_id, scheduler::RequestId{21});
  run_to_completion(engine, kStart + 3ms);
  EXPECT_EQ(engine.stats().kv_cache.block_reuse_count, 1U);
}

TEST(CpuRuntimeEngineTest, AppliesEosAndCancellationWithCompleteKvReclamation) {
  const model::LlamaConfig model_config = runtime_model_config();
  model::CpuLlamaModel reference = make_model(model_config);
  const std::vector<std::uint32_t> prompt{4, 2};
  const scheduler::TokenId eos(model::select_greedy_token(reference.forward(prompt)));
  CpuRuntimeEngine engine(make_model(model_config),
                          CpuRuntimeConfig{scheduler::SchedulerConfig{4, 2, 2}, 6});
  engine.submit(make_request(30, prompt, 4), eos);
  static_cast<void>(engine.run_iteration(kStart));
  const CpuRuntimeIterationResult eos_iteration = engine.run_iteration(kStart + 1ms);
  ASSERT_EQ(eos_iteration.retired_requests.size(), 1U);
  const scheduler::Request& eos_request = engine.request(scheduler::RequestId{30});
  EXPECT_EQ(eos_request.finish_reason(), scheduler::FinishReason::kEndOfSequence);
  ASSERT_EQ(eos_request.generated_tokens().size(), 1U);
  EXPECT_EQ(eos_request.generated_tokens()[0].id, eos);

  engine.submit(make_request(31, {1, 2, 3}, 2, kStart + 2ms));
  static_cast<void>(engine.run_iteration(kStart + 2ms));
  EXPECT_GT(engine.stats().kv_cache.used_blocks, 0U);
  const scheduler::RetiredRequest cancelled =
      engine.cancel(scheduler::RequestId{31}, kStart + 3ms, "client disconnected");
  EXPECT_EQ(cancelled.state, scheduler::RequestState::kCancelled);
  EXPECT_EQ(engine.request(scheduler::RequestId{31}).terminal_context(), "client disconnected");

  engine.submit(make_request(32, {1}, 1, kStart + 4ms));
  static_cast<void>(engine.cancel(scheduler::RequestId{32}, kStart + 4ms));
  const CpuRuntimeStats stats = engine.stats();
  EXPECT_EQ(stats.active_session_count, 0U);
  EXPECT_EQ(stats.kv_cache.active_sequence_count, 0U);
  EXPECT_EQ(stats.kv_cache.free_blocks, stats.kv_cache.total_blocks);
}

TEST(CpuRuntimeEngineTest, SustainsMixedRequestsUnderKvPressureWithoutLeaks) {
  constexpr std::size_t kRequestCount = 32;
  const model::LlamaConfig model_config = runtime_model_config();
  model::CpuLlamaModel reference = make_model(model_config);
  CpuRuntimeEngine engine(make_model(model_config),
                          CpuRuntimeConfig{scheduler::SchedulerConfig{8, 4, 2}, 8});

  std::vector<std::vector<std::uint32_t>> expected;
  expected.reserve(kRequestCount);
  for (std::size_t index = 0; index < kRequestCount; ++index) {
    const std::size_t prompt_length = 1 + index % 3;
    const std::size_t output_length = 1 + index % 4;
    std::vector<std::uint32_t> prompt;
    prompt.reserve(prompt_length);
    for (std::size_t position = 0; position < prompt_length; ++position) {
      prompt.push_back(
          static_cast<std::uint32_t>((index * 3 + position * 5) % model_config.vocabulary_size));
    }
    expected.push_back(full_recompute_greedy(reference, prompt, output_length));
    engine.submit(make_request(100 + index, std::move(prompt), output_length));
  }

  bool observed_kv_pressure = false;
  std::size_t iterations = 0;
  for (; iterations < 256 && engine.has_live_requests(); ++iterations) {
    const CpuRuntimeIterationResult result =
        engine.run_iteration(kStart + std::chrono::milliseconds(iterations));
    ASSERT_TRUE(result.iteration_id.has_value()) << "runtime made no progress at " << iterations;
    observed_kv_pressure = observed_kv_pressure || result.kv_capacity_limited;
  }

  EXPECT_FALSE(engine.has_live_requests());
  EXPECT_LT(iterations, 256U);
  EXPECT_TRUE(observed_kv_pressure);
  for (std::size_t index = 0; index < kRequestCount; ++index) {
    const scheduler::Request& completed = engine.request(scheduler::RequestId{100 + index});
    EXPECT_EQ(completed.state(), scheduler::RequestState::kFinished) << "request " << index;
    EXPECT_EQ(completed.finish_reason(), scheduler::FinishReason::kLength) << "request " << index;
    EXPECT_EQ(generated_values(completed), expected[index]) << "request " << index;
  }

  const CpuRuntimeStats stats = engine.stats();
  EXPECT_EQ(stats.scheduler.submitted_count, kRequestCount);
  EXPECT_EQ(stats.scheduler.finished_count, kRequestCount);
  EXPECT_EQ(stats.scheduler.cancelled_count, 0U);
  EXPECT_EQ(stats.scheduler.failed_count, 0U);
  EXPECT_EQ(stats.active_session_count, 0U);
  EXPECT_EQ(stats.kv_cache.active_sequence_count, 0U);
  EXPECT_EQ(stats.kv_cache.active_tokens, 0U);
  EXPECT_EQ(stats.kv_cache.used_blocks, 0U);
  EXPECT_EQ(stats.kv_cache.free_blocks, stats.kv_cache.total_blocks);
  EXPECT_GT(stats.kv_cache.block_reuse_count, 0U);
}

TEST(CpuRuntimeEngineTest, RejectsInvalidSubmissionsWithoutLeakingSessions) {
  const model::LlamaConfig model_config = runtime_model_config(6);
  EXPECT_THROW(
      static_cast<void>(CpuRuntimeEngine(make_model(model_config),
                                         CpuRuntimeConfig{scheduler::SchedulerConfig{4, 2, 2}, 2,
                                                          static_cast<CpuPrefillMode>(255)})),
      Error);
  EXPECT_THROW(
      static_cast<void>(CpuRuntimeEngine(
          make_model(model_config),
          CpuRuntimeConfig{scheduler::SchedulerConfig{4, 2, 2}, 2, CpuPrefillMode::kFullPrompt,
                           static_cast<model::PagedDecodeMode>(255)})),
      Error);
  CpuRuntimeEngine engine(make_model(model_config),
                          CpuRuntimeConfig{scheduler::SchedulerConfig{4, 2, 2}, 2});

  EXPECT_THROW(engine.submit(make_request(40, {11}, 1)), Error);
  EXPECT_THROW(engine.submit(make_request(41, {1, 2, 3, 4, 5}, 2)), Error);
  EXPECT_THROW(engine.submit(make_request(42, {1, 2, 3}, 3)), Error);
  EXPECT_THROW(engine.submit(make_request(43, {1}, 1), scheduler::TokenId{11}), Error);
  EXPECT_FALSE(engine.contains(scheduler::RequestId{40}));

  engine.submit(make_request(44, {1}, 1));
  EXPECT_THROW(engine.submit(make_request(44, {2}, 1)), Error);
  const CpuRuntimeStats stats = engine.stats();
  EXPECT_EQ(stats.scheduler.request_count, 1U);
  EXPECT_EQ(stats.active_session_count, 1U);
  EXPECT_EQ(stats.kv_cache.active_sequence_count, 1U);
  EXPECT_EQ(stats.kv_cache.used_blocks, 0U);
}

TEST(CpuRuntimeEngineTest, ConvertsModelFailureIntoFailedRequestAndReclaimsKv) {
  const model::LlamaConfig model_config = runtime_model_config();
  Tensor lm_head_alias = Tensor::zeros({1});
  CpuRuntimeEngine engine(make_model(model_config, &lm_head_alias),
                          CpuRuntimeConfig{scheduler::SchedulerConfig{4, 1, 2}, 4});
  engine.submit(make_request(50, {1, 2}, 2));
  lm_head_alias.data_as<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  static_cast<void>(engine.run_iteration(kStart));

  const CpuRuntimeIterationResult failed = engine.run_iteration(kStart + 1ms);
  ASSERT_EQ(failed.completions.size(), 1U);
  EXPECT_EQ(failed.completions[0].status, scheduler::WorkStatus::kFailed);
  ASSERT_EQ(failed.retired_requests.size(), 1U);
  EXPECT_EQ(engine.request(scheduler::RequestId{50}).state(), scheduler::RequestState::kFailed);
  EXPECT_NE(engine.request(scheduler::RequestId{50}).terminal_context().find("non-finite"),
            std::string_view::npos);

  const CpuRuntimeStats stats = engine.stats();
  EXPECT_EQ(stats.failed_work_item_count, 1U);
  EXPECT_EQ(stats.active_session_count, 0U);
  EXPECT_EQ(stats.kv_cache.used_blocks, 0U);
  EXPECT_FALSE(engine.has_live_requests());
}

} // namespace
} // namespace helix::runtime
