#include "helix/core/error.hpp"
#include "helix/scheduler/fifo_scheduler.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <vector>

namespace helix::scheduler {
namespace {

using namespace std::chrono_literals;

constexpr RequestTimePoint kStart{};

std::vector<TokenId> prompt(std::size_t size) {
  std::vector<TokenId> result;
  result.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    result.emplace_back(static_cast<std::uint32_t>(1'000 + index));
  }
  return result;
}

Request make_request(std::uint64_t id, std::size_t prompt_size, std::size_t max_new_tokens,
                     RequestTimePoint created_at = kStart) {
  return Request(RequestId{id}, prompt(prompt_size), max_new_tokens, created_at);
}

WorkCompletion prefill_success(RequestId id) {
  return WorkCompletion{id, WorkStatus::kSucceeded, std::nullopt, false, {}};
}

WorkCompletion decode_success(RequestId id, std::uint32_t token, bool end_of_sequence = false) {
  return WorkCompletion{id, WorkStatus::kSucceeded, TokenId{token}, end_of_sequence, {}};
}

TEST(FifoSchedulerTest, ValidatesConfigurationAndWorkKindNames) {
  EXPECT_THROW(static_cast<void>(FifoScheduler(SchedulerConfig{0, 1, 1})), Error);
  EXPECT_THROW(static_cast<void>(FifoScheduler(SchedulerConfig{1, 0, 1})), Error);
  EXPECT_THROW(static_cast<void>(FifoScheduler(SchedulerConfig{1, 1, 0})), Error);

  FifoScheduler scheduler(SchedulerConfig{7, 3, 4});
  EXPECT_EQ(scheduler.config().max_tokens_per_iteration, 7U);
  EXPECT_EQ(work_kind_name(WorkKind::kPrefill), "prefill");
  EXPECT_EQ(work_kind_name(WorkKind::kDecode), "decode");
  EXPECT_THROW(static_cast<void>(work_kind_name(static_cast<WorkKind>(255))), Error);
}

TEST(FifoSchedulerTest, StrictFifoBuildsMixedPrefillAndDecodeWithinTokenBudget) {
  FifoScheduler scheduler(SchedulerConfig{4, 3, 4});
  scheduler.submit(make_request(1, 3, 2));
  scheduler.submit(make_request(2, 4, 2));
  scheduler.submit(make_request(3, 1, 1));

  const ScheduleResult first = scheduler.schedule_iteration(10, kStart + 1ms);
  ASSERT_TRUE(first.batch.has_value());
  EXPECT_TRUE(first.token_budget_exhausted);
  EXPECT_FALSE(first.kv_capacity_limited);
  EXPECT_EQ(first.batch->scheduled_tokens, 4U);
  EXPECT_EQ(first.batch->new_kv_blocks, 2U);
  ASSERT_EQ(first.batch->work.size(), 2U);
  EXPECT_EQ(first.batch->work[0].request_id, RequestId{1});
  EXPECT_EQ(first.batch->work[0].kind, WorkKind::kPrefill);
  EXPECT_EQ(first.batch->work[0].token_count, 3U);
  EXPECT_EQ(first.batch->work[1].request_id, RequestId{2});
  EXPECT_EQ(first.batch->work[1].token_count, 1U);
  EXPECT_EQ(scheduler.waiting_requests(), (std::vector{RequestId{3}}));
  EXPECT_EQ(scheduler.running_requests().size(), 2U);

  const std::vector first_completions{prefill_success(RequestId{1}), prefill_success(RequestId{2})};
  const CompletionReport first_report =
      scheduler.complete_iteration(first.batch->id, first_completions, kStart + 2ms);
  EXPECT_TRUE(first_report.retired_requests.empty());
  EXPECT_EQ(scheduler.request(RequestId{1}).state(), RequestState::kDecoding);
  EXPECT_EQ(scheduler.request(RequestId{2}).state(), RequestState::kPrefill);

  const ScheduleResult second = scheduler.schedule_iteration(8, kStart + 3ms);
  ASSERT_TRUE(second.batch.has_value());
  ASSERT_EQ(second.batch->work.size(), 2U);
  EXPECT_EQ(second.batch->work[0].request_id, RequestId{1});
  EXPECT_EQ(second.batch->work[0].kind, WorkKind::kDecode);
  EXPECT_EQ(second.batch->work[0].token_count, 1U);
  EXPECT_EQ(second.batch->work[0].new_kv_blocks, 0U);
  EXPECT_EQ(second.batch->work[1].request_id, RequestId{2});
  EXPECT_EQ(second.batch->work[1].kind, WorkKind::kPrefill);
  EXPECT_EQ(second.batch->work[1].token_offset, 1U);
  EXPECT_EQ(second.batch->work[1].token_count, 3U);

  const std::vector second_completions{decode_success(RequestId{1}, 91),
                                       prefill_success(RequestId{2})};
  static_cast<void>(
      scheduler.complete_iteration(second.batch->id, second_completions, kStart + 4ms));
  EXPECT_EQ(scheduler.request(RequestId{1}).generated_tokens().size(), 1U);
  EXPECT_EQ(scheduler.request(RequestId{2}).state(), RequestState::kDecoding);
  EXPECT_EQ(scheduler.waiting_requests(), (std::vector{RequestId{3}}));
}

TEST(FifoSchedulerTest, KvExhaustionChunksPrefillAndUsesExistingBlockSlack) {
  FifoScheduler scheduler(SchedulerConfig{8, 1, 4});
  scheduler.submit(make_request(10, 6, 1));

  const ScheduleResult first = scheduler.schedule_iteration(1, kStart);
  ASSERT_TRUE(first.batch.has_value());
  ASSERT_EQ(first.batch->work.size(), 1U);
  EXPECT_EQ(first.batch->work[0].token_count, 4U);
  EXPECT_EQ(first.batch->work[0].new_kv_blocks, 1U);
  EXPECT_TRUE(first.kv_capacity_limited);
  EXPECT_EQ(first.unused_token_budget, 4U);
  static_cast<void>(scheduler.complete_iteration(
      first.batch->id, std::vector{prefill_success(RequestId{10})}, kStart + 1ms));

  const ScheduleResult blocked = scheduler.schedule_iteration(0, kStart + 2ms);
  EXPECT_FALSE(blocked.batch.has_value());
  EXPECT_TRUE(blocked.kv_capacity_limited);
  EXPECT_EQ(scheduler.request(RequestId{10}).prompt_tokens_processed(), 4U);

  const ScheduleResult second = scheduler.schedule_iteration(1, kStart + 3ms);
  ASSERT_TRUE(second.batch.has_value());
  EXPECT_EQ(second.batch->work[0].token_offset, 4U);
  EXPECT_EQ(second.batch->work[0].token_count, 2U);
  EXPECT_EQ(second.batch->work[0].new_kv_blocks, 1U);
  static_cast<void>(scheduler.complete_iteration(
      second.batch->id, std::vector{prefill_success(RequestId{10})}, kStart + 4ms));
  EXPECT_EQ(scheduler.request(RequestId{10}).state(), RequestState::kDecoding);

  const ScheduleResult decode = scheduler.schedule_iteration(0, kStart + 5ms);
  ASSERT_TRUE(decode.batch.has_value());
  EXPECT_EQ(decode.batch->work[0].kind, WorkKind::kDecode);
  EXPECT_EQ(decode.batch->work[0].new_kv_blocks, 0U);
  const CompletionReport report = scheduler.complete_iteration(
      decode.batch->id, std::vector{decode_success(RequestId{10}, 777)}, kStart + 6ms);
  ASSERT_EQ(report.retired_requests.size(), 1U);
  EXPECT_EQ(report.retired_requests[0].state, RequestState::kFinished);
  EXPECT_EQ(report.released_kv_blocks, 2U);
  EXPECT_TRUE(scheduler.running_requests().empty());
}

TEST(FifoSchedulerTest, FinalDecodeNeedsNoNewBlockAtAnExactBoundary) {
  FifoScheduler scheduler(SchedulerConfig{4, 1, 2});
  scheduler.submit(make_request(11, 2, 1));

  const ScheduleResult prefill = scheduler.schedule_iteration(1, kStart);
  ASSERT_TRUE(prefill.batch.has_value());
  ASSERT_EQ(prefill.batch->work.size(), 1U);
  EXPECT_EQ(prefill.batch->work[0].new_kv_blocks, 1U);
  static_cast<void>(scheduler.complete_iteration(
      prefill.batch->id, std::vector{prefill_success(RequestId{11})}, kStart + 1ms));

  const ScheduleResult final_decode = scheduler.schedule_iteration(0, kStart + 2ms);
  ASSERT_TRUE(final_decode.batch.has_value());
  ASSERT_EQ(final_decode.batch->work.size(), 1U);
  EXPECT_EQ(final_decode.batch->work[0].kind, WorkKind::kDecode);
  EXPECT_EQ(final_decode.batch->work[0].kv_blocks_before, 1U);
  EXPECT_EQ(final_decode.batch->work[0].new_kv_blocks, 0U);
  EXPECT_FALSE(final_decode.kv_capacity_limited);

  const CompletionReport completion = scheduler.complete_iteration(
      final_decode.batch->id, std::vector{decode_success(RequestId{11}, 778)}, kStart + 3ms);
  EXPECT_EQ(completion.released_kv_blocks, 1U);
  EXPECT_EQ(scheduler.request(RequestId{11}).finish_reason(), FinishReason::kLength);
}

TEST(FifoSchedulerTest, RunningCapacityLeavesLaterRequestsInFifoOrder) {
  FifoScheduler scheduler(SchedulerConfig{16, 1, 4});
  scheduler.submit(make_request(20, 1, 2));
  scheduler.submit(make_request(21, 1, 2));

  const ScheduleResult result = scheduler.schedule_iteration(4, kStart);
  ASSERT_TRUE(result.batch.has_value());
  ASSERT_EQ(result.batch->work.size(), 1U);
  EXPECT_EQ(result.batch->work[0].request_id, RequestId{20});
  EXPECT_EQ(scheduler.waiting_requests(), (std::vector{RequestId{21}}));
  ASSERT_EQ(scheduler.running_requests().size(), 1U);
  EXPECT_EQ(scheduler.running_requests()[0], RequestId{20});
}

TEST(FifoSchedulerTest, CompletionValidationRejectsMalformedResultsWithoutProgress) {
  FifoScheduler scheduler(SchedulerConfig{4, 1, 4});
  scheduler.submit(make_request(30, 1, 1));
  const ScheduleResult prefill = scheduler.schedule_iteration(1, kStart);
  ASSERT_TRUE(prefill.batch.has_value());

  EXPECT_THROW(static_cast<void>(scheduler.complete_iteration(
                   IterationId{999}, std::vector{prefill_success(RequestId{30})}, kStart + 1ms)),
               Error);
  EXPECT_THROW(static_cast<void>(scheduler.complete_iteration(prefill.batch->id, {}, kStart + 1ms)),
               Error);
  EXPECT_THROW(static_cast<void>(scheduler.complete_iteration(
                   prefill.batch->id, std::vector{decode_success(RequestId{30}, 8)}, kStart + 1ms)),
               Error);
  EXPECT_TRUE(scheduler.iteration_in_flight());
  EXPECT_EQ(scheduler.request(RequestId{30}).prompt_tokens_processed(), 0U);

  static_cast<void>(scheduler.complete_iteration(
      prefill.batch->id, std::vector{prefill_success(RequestId{30})}, kStart + 1ms));
  const ScheduleResult decode = scheduler.schedule_iteration(0, kStart + 2ms);
  ASSERT_TRUE(decode.batch.has_value());
  EXPECT_THROW(static_cast<void>(scheduler.complete_iteration(
                   decode.batch->id, std::vector{prefill_success(RequestId{30})}, kStart + 3ms)),
               Error);
  EXPECT_TRUE(scheduler.request(RequestId{30}).generated_tokens().empty());

  const CompletionReport report = scheduler.complete_iteration(
      decode.batch->id, std::vector{decode_success(RequestId{30}, 9)}, kStart + 3ms);
  EXPECT_EQ(report.retired_requests.size(), 1U);
  EXPECT_EQ(scheduler.request(RequestId{30}).state(), RequestState::kFinished);
}

TEST(FifoSchedulerTest, ValidatesWholeCompletionVectorBeforeMutatingAnyRequest) {
  FifoScheduler scheduler(SchedulerConfig{4, 2, 4});
  scheduler.submit(make_request(31, 1, 1));
  scheduler.submit(make_request(32, 1, 1));
  const ScheduleResult batch = scheduler.schedule_iteration(2, kStart);
  ASSERT_TRUE(batch.batch.has_value());
  ASSERT_EQ(batch.batch->work.size(), 2U);

  const std::vector malformed{prefill_success(RequestId{31}), decode_success(RequestId{32}, 99)};
  EXPECT_THROW(
      static_cast<void>(scheduler.complete_iteration(batch.batch->id, malformed, kStart + 1ms)),
      Error);
  EXPECT_EQ(scheduler.request(RequestId{31}).prompt_tokens_processed(), 0U);
  EXPECT_EQ(scheduler.request(RequestId{32}).prompt_tokens_processed(), 0U);
  EXPECT_TRUE(scheduler.iteration_in_flight());

  const std::vector valid{prefill_success(RequestId{31}), prefill_success(RequestId{32})};
  static_cast<void>(scheduler.complete_iteration(batch.batch->id, valid, kStart + 1ms));
  EXPECT_EQ(scheduler.request(RequestId{31}).state(), RequestState::kDecoding);
  EXPECT_EQ(scheduler.request(RequestId{32}).state(), RequestState::kDecoding);
}

TEST(FifoSchedulerTest, FailedWorkRetiresReservedBlocksAndPreservesError) {
  FifoScheduler scheduler(SchedulerConfig{8, 1, 4});
  scheduler.submit(make_request(40, 5, 4));
  const ScheduleResult batch = scheduler.schedule_iteration(2, kStart);
  ASSERT_TRUE(batch.batch.has_value());
  ASSERT_EQ(batch.batch->work[0].kv_blocks_after(), 2U);

  const WorkCompletion failure{RequestId{40}, WorkStatus::kFailed, std::nullopt, false,
                               "model executor failed"};
  const CompletionReport report =
      scheduler.complete_iteration(batch.batch->id, std::vector{failure}, kStart + 1ms);
  ASSERT_EQ(report.retired_requests.size(), 1U);
  EXPECT_EQ(report.released_kv_blocks, 2U);
  EXPECT_EQ(scheduler.request(RequestId{40}).state(), RequestState::kFailed);
  EXPECT_EQ(scheduler.request(RequestId{40}).terminal_context(), "model executor failed");
  EXPECT_TRUE(scheduler.running_requests().empty());
}

TEST(FifoSchedulerTest, CancellationHandlesWaitingAndRunningOwnership) {
  FifoScheduler scheduler(SchedulerConfig{4, 1, 4});
  scheduler.submit(make_request(50, 3, 2));
  scheduler.submit(make_request(51, 2, 2));

  const RetiredRequest waiting = scheduler.cancel(RequestId{51}, kStart, "cancelled while queued");
  EXPECT_EQ(waiting.released_kv_blocks, 0U);
  EXPECT_EQ(scheduler.request(RequestId{51}).state(), RequestState::kCancelled);
  EXPECT_TRUE(scheduler.waiting_requests().empty() ||
              scheduler.waiting_requests().front() != RequestId{51});

  const ScheduleResult batch = scheduler.schedule_iteration(1, kStart + 1ms);
  ASSERT_TRUE(batch.batch.has_value());
  EXPECT_THROW(static_cast<void>(scheduler.cancel(RequestId{50}, kStart + 1ms)), Error);
  static_cast<void>(scheduler.complete_iteration(
      batch.batch->id, std::vector{prefill_success(RequestId{50})}, kStart + 2ms));
  const RetiredRequest running = scheduler.cancel(RequestId{50}, kStart + 3ms);
  EXPECT_EQ(running.released_kv_blocks, 1U);
  EXPECT_EQ(scheduler.request(RequestId{50}).state(), RequestState::kCancelled);
  EXPECT_TRUE(scheduler.running_requests().empty());
}

TEST(FifoSchedulerTest, SubmissionLookupAndTimeErrorsAreContextualAndTransactional) {
  FifoScheduler scheduler(SchedulerConfig{4, 2, 4});
  scheduler.submit(make_request(60, 1, 1, kStart + 10ms));
  EXPECT_THROW(scheduler.submit(make_request(60, 1, 1)), Error);
  EXPECT_THROW(static_cast<void>(scheduler.request(RequestId{999})), Error);
  EXPECT_THROW(static_cast<void>(scheduler.cancel(RequestId{999}, kStart)), Error);

  EXPECT_THROW(static_cast<void>(scheduler.schedule_iteration(1, kStart + 5ms)), Error);
  EXPECT_EQ(scheduler.request(RequestId{60}).state(), RequestState::kWaiting);

  const ScheduleResult batch = scheduler.schedule_iteration(1, kStart + 10ms);
  ASSERT_TRUE(batch.batch.has_value());
  EXPECT_THROW(static_cast<void>(scheduler.complete_iteration(
                   batch.batch->id, std::vector{prefill_success(RequestId{60})}, kStart + 9ms)),
               Error);
  EXPECT_EQ(scheduler.request(RequestId{60}).prompt_tokens_processed(), 0U);
}

TEST(FifoSchedulerTest, StatisticsSeparateLiveQueuesAndTerminalHistory) {
  FifoScheduler scheduler(SchedulerConfig{4, 2, 4});
  scheduler.submit(make_request(70, 1, 0));
  scheduler.submit(make_request(71, 1, 1));
  EXPECT_EQ(scheduler.stats().submitted_count, 2U);
  EXPECT_EQ(scheduler.stats().waiting_count, 2U);

  const ScheduleResult batch = scheduler.schedule_iteration(2, kStart);
  ASSERT_TRUE(batch.batch.has_value());
  const std::vector completions{prefill_success(RequestId{70}), prefill_success(RequestId{71})};
  const CompletionReport report =
      scheduler.complete_iteration(batch.batch->id, completions, kStart + 1ms);
  ASSERT_EQ(report.retired_requests.size(), 1U);

  const RetiredRequest failed = scheduler.fail(RequestId{71}, kStart + 2ms, "capacity removed");
  EXPECT_EQ(failed.state, RequestState::kFailed);
  const SchedulerStats stats = scheduler.stats();
  EXPECT_EQ(stats.request_count, 2U);
  EXPECT_EQ(stats.waiting_count, 0U);
  EXPECT_EQ(stats.running_count, 0U);
  EXPECT_EQ(stats.finished_count, 1U);
  EXPECT_EQ(stats.failed_count, 1U);
  EXPECT_EQ(stats.scheduled_iteration_count, 1U);
  EXPECT_EQ(stats.completed_iteration_count, 1U);
  EXPECT_EQ(stats.scheduled_prefill_tokens, 2U);
  EXPECT_EQ(stats.scheduled_decode_tokens, 0U);

  const SchedulerMetrics metrics = scheduler.metrics(kStart + 2ms);
  EXPECT_EQ(metrics.finished_count, 1U);
  EXPECT_EQ(metrics.failed_count, 1U);
  EXPECT_EQ(metrics.request_latency.sample_count, 1U);
}

TEST(FifoSchedulerTest, FinishedRequestIsReplacedBeforeOriginalCohortDrains) {
  FifoScheduler scheduler(SchedulerConfig{16, 4, 4});
  scheduler.submit(make_request(80, 1, 1));
  scheduler.submit(make_request(81, 1, 3));
  scheduler.submit(make_request(82, 1, 3));
  scheduler.submit(make_request(83, 1, 3));
  scheduler.submit(make_request(84, 1, 1));

  std::size_t available_blocks = 20;
  const ScheduleResult prefill = scheduler.schedule_iteration(available_blocks, kStart);
  ASSERT_TRUE(prefill.batch.has_value());
  ASSERT_EQ(prefill.batch->work.size(), 4U);
  EXPECT_EQ(prefill.batch->work[0].request_id, RequestId{80});
  EXPECT_EQ(prefill.batch->work[3].request_id, RequestId{83});
  available_blocks -= prefill.batch->new_kv_blocks;
  const std::vector prefill_completions{
      prefill_success(RequestId{80}), prefill_success(RequestId{81}),
      prefill_success(RequestId{82}), prefill_success(RequestId{83})};
  static_cast<void>(
      scheduler.complete_iteration(prefill.batch->id, prefill_completions, kStart + 1ms));

  const ScheduleResult first_decode = scheduler.schedule_iteration(available_blocks, kStart + 2ms);
  ASSERT_TRUE(first_decode.batch.has_value());
  ASSERT_EQ(first_decode.batch->work.size(), 4U);
  EXPECT_THROW(static_cast<void>(scheduler.metrics(kStart + 1ms)), Error);
  const std::vector decode_completions{
      decode_success(RequestId{80}, 180), decode_success(RequestId{81}, 181),
      decode_success(RequestId{82}, 182), decode_success(RequestId{83}, 183)};
  const CompletionReport completed =
      scheduler.complete_iteration(first_decode.batch->id, decode_completions, kStart + 3ms);
  ASSERT_EQ(completed.retired_requests.size(), 1U);
  EXPECT_EQ(completed.retired_requests[0].request_id, RequestId{80});
  available_blocks += completed.released_kv_blocks;

  const ScheduleResult replacement = scheduler.schedule_iteration(available_blocks, kStart + 4ms);
  ASSERT_TRUE(replacement.batch.has_value());
  ASSERT_EQ(replacement.batch->work.size(), 4U);
  EXPECT_EQ(replacement.batch->work[0].request_id, RequestId{81});
  EXPECT_EQ(replacement.batch->work[1].request_id, RequestId{82});
  EXPECT_EQ(replacement.batch->work[2].request_id, RequestId{83});
  EXPECT_EQ(replacement.batch->work[3].request_id, RequestId{84});
  EXPECT_EQ(replacement.batch->work[3].kind, WorkKind::kPrefill);
  EXPECT_TRUE(scheduler.waiting_requests().empty());
  ASSERT_EQ(scheduler.running_requests().size(), 4U);
  EXPECT_EQ(scheduler.running_requests()[0], RequestId{81});
  EXPECT_EQ(scheduler.running_requests()[1], RequestId{82});
  EXPECT_EQ(scheduler.running_requests()[2], RequestId{83});
  EXPECT_EQ(scheduler.running_requests()[3], RequestId{84});
  EXPECT_EQ(scheduler.request(RequestId{81}).state(), RequestState::kDecoding);
  EXPECT_EQ(scheduler.request(RequestId{84}).state(), RequestState::kPrefill);
}

} // namespace
} // namespace helix::scheduler
