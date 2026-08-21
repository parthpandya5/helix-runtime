#include "helix/core/error.hpp"
#include "helix/scheduler/metrics.hpp"

#include <chrono>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace helix::scheduler {
namespace {

using namespace std::chrono_literals;

constexpr RequestTimePoint kStart{};

std::vector<TokenId> single_token_prompt(std::uint32_t token) {
  return std::vector{TokenId{token}};
}

TEST(SchedulerMetricsTest, AggregatesRatesAndInterpolatedLatencyStatistics) {
  Request first(RequestId{1}, single_token_prompt(11), 2, kStart);
  first.begin_prefill(kStart + 1ms);
  first.record_prefill_progress(1, kStart + 2ms);
  first.append_generated_token(TokenId{21}, kStart + 3ms);
  first.append_generated_token(TokenId{22}, kStart + 5ms);

  Request second(RequestId{2}, single_token_prompt(12), 0, kStart);
  second.begin_prefill(kStart + 2ms);
  second.record_prefill_progress(1, kStart + 3ms);

  Request cancelled(RequestId{3}, single_token_prompt(13), 4, kStart + 1ms);
  cancelled.cancel(kStart + 4ms, "caller departed");

  const std::vector<const Request*> requests{&first, &second, &cancelled};
  const SchedulerMetrics metrics = summarize_request_metrics(requests, kStart + 10ms);

  EXPECT_EQ(metrics.request_count, 3U);
  EXPECT_EQ(metrics.finished_count, 2U);
  EXPECT_EQ(metrics.cancelled_count, 1U);
  EXPECT_EQ(metrics.waiting_count, 0U);
  EXPECT_EQ(metrics.prefill_count, 0U);
  EXPECT_EQ(metrics.decoding_count, 0U);
  EXPECT_EQ(metrics.failed_count, 0U);
  EXPECT_EQ(metrics.prompt_tokens_processed, 2U);
  EXPECT_EQ(metrics.generated_tokens, 2U);
  EXPECT_DOUBLE_EQ(metrics.observation_seconds, 0.010);
  EXPECT_TRUE(metrics.throughput_defined);
  EXPECT_DOUBLE_EQ(metrics.finished_requests_per_second, 200.0);
  EXPECT_DOUBLE_EQ(metrics.generated_tokens_per_second, 200.0);
  EXPECT_DOUBLE_EQ(metrics.processed_tokens_per_second, 400.0);

  EXPECT_EQ(metrics.waiting_time.sample_count, 2U);
  EXPECT_DOUBLE_EQ(metrics.waiting_time.minimum_seconds, 0.001);
  EXPECT_DOUBLE_EQ(metrics.waiting_time.mean_seconds, 0.0015);
  EXPECT_DOUBLE_EQ(metrics.waiting_time.p50_seconds, 0.0015);
  EXPECT_DOUBLE_EQ(metrics.waiting_time.p95_seconds, 0.00195);
  EXPECT_DOUBLE_EQ(metrics.waiting_time.maximum_seconds, 0.002);

  EXPECT_EQ(metrics.time_to_first_token.sample_count, 1U);
  EXPECT_DOUBLE_EQ(metrics.time_to_first_token.p50_seconds, 0.003);
  EXPECT_EQ(metrics.request_latency.sample_count, 2U);
  EXPECT_DOUBLE_EQ(metrics.request_latency.mean_seconds, 0.004);
  EXPECT_DOUBLE_EQ(metrics.request_latency.p50_seconds, 0.004);
  EXPECT_DOUBLE_EQ(metrics.request_latency.p95_seconds, 0.0049);
  EXPECT_EQ(metrics.inter_token_latency.sample_count, 1U);
  EXPECT_DOUBLE_EQ(metrics.inter_token_latency.p50_seconds, 0.002);
}

TEST(SchedulerMetricsTest, CountsActiveStatesAndUsesExplicitObservationBoundary) {
  Request waiting(RequestId{10}, single_token_prompt(1), 1, kStart);

  Request prefill(RequestId{11}, std::vector{TokenId{1}, TokenId{2}}, 1, kStart);
  prefill.begin_prefill(kStart + 1ms);
  prefill.record_prefill_progress(1, kStart + 2ms);

  Request decoding(RequestId{12}, single_token_prompt(3), 2, kStart);
  decoding.begin_prefill(kStart + 1ms);
  decoding.record_prefill_progress(1, kStart + 2ms);
  decoding.append_generated_token(TokenId{4}, kStart + 3ms);

  Request failed(RequestId{13}, single_token_prompt(5), 1, kStart);
  failed.fail(kStart + 1ms, "model error");

  const std::vector<const Request*> requests{&waiting, &prefill, &decoding, &failed};
  const SchedulerMetrics metrics = summarize_request_metrics(requests, kStart + 4ms);
  EXPECT_EQ(metrics.waiting_count, 1U);
  EXPECT_EQ(metrics.prefill_count, 1U);
  EXPECT_EQ(metrics.decoding_count, 1U);
  EXPECT_EQ(metrics.failed_count, 1U);
  EXPECT_EQ(metrics.finished_count, 0U);
  EXPECT_EQ(metrics.prompt_tokens_processed, 2U);
  EXPECT_EQ(metrics.generated_tokens, 1U);
  EXPECT_EQ(metrics.waiting_time.sample_count, 2U);
  EXPECT_EQ(metrics.time_to_first_token.sample_count, 1U);
  EXPECT_EQ(metrics.request_latency.sample_count, 0U);
  EXPECT_EQ(metrics.inter_token_latency.sample_count, 0U);
}

TEST(SchedulerMetricsTest, RejectsNullDuplicateAndRegressiveSnapshots) {
  EXPECT_FALSE(summarize_request_metrics({}, kStart).throughput_defined);

  Request request(RequestId{20}, single_token_prompt(1), 1, kStart);
  request.begin_prefill(kStart + 2ms);
  const std::vector<const Request*> valid{&request};
  EXPECT_THROW(static_cast<void>(summarize_request_metrics(valid, kStart + 1ms)), Error);

  const std::vector<const Request*> duplicate{&request, &request};
  EXPECT_THROW(static_cast<void>(summarize_request_metrics(duplicate, kStart + 2ms)), Error);

  const std::vector<const Request*> null_request{nullptr};
  EXPECT_THROW(static_cast<void>(summarize_request_metrics(null_request, kStart)), Error);

  Request moved_from(RequestId{22}, single_token_prompt(3), 1, kStart);
  Request moved_to(std::move(moved_from));
  const std::vector<const Request*> invalid_id{&moved_from};
  EXPECT_THROW(static_cast<void>(summarize_request_metrics(invalid_id, kStart)), Error);
  EXPECT_EQ(moved_to.id(), RequestId{22});

  Request zero_window(RequestId{21}, single_token_prompt(2), 1, kStart);
  const std::vector<const Request*> one_waiting{&zero_window};
  const SchedulerMetrics zero = summarize_request_metrics(one_waiting, kStart);
  EXPECT_FALSE(zero.throughput_defined);
  EXPECT_DOUBLE_EQ(zero.observation_seconds, 0.0);
}

} // namespace
} // namespace helix::scheduler
