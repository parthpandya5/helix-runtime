#pragma once

#include "helix/scheduler/request.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace helix::scheduler {

struct DurationStatistics {
  std::size_t sample_count = 0;
  double minimum_seconds = 0.0;
  double mean_seconds = 0.0;
  double p50_seconds = 0.0;
  double p95_seconds = 0.0;
  double maximum_seconds = 0.0;
};

struct SchedulerMetrics {
  std::size_t request_count = 0;
  std::size_t waiting_count = 0;
  std::size_t prefill_count = 0;
  std::size_t decoding_count = 0;
  std::size_t finished_count = 0;
  std::size_t cancelled_count = 0;
  std::size_t failed_count = 0;

  std::uint64_t prompt_tokens_processed = 0;
  std::uint64_t generated_tokens = 0;

  double observation_seconds = 0.0;
  bool throughput_defined = false;
  double finished_requests_per_second = 0.0;
  double generated_tokens_per_second = 0.0;
  double processed_tokens_per_second = 0.0;

  DurationStatistics waiting_time;
  DurationStatistics time_to_first_token;
  DurationStatistics request_latency;
  DurationStatistics inter_token_latency;
};

// Summarizes immutable request state at an explicit observation boundary. P50 and
// P95 use linear interpolation over sorted samples at p*(N-1). Request latency
// includes successfully finished requests only; cancellation and failure retain
// separate counts. Rates are defined only for a positive window beginning at the
// earliest request creation time.
[[nodiscard]] SchedulerMetrics summarize_request_metrics(std::span<const Request* const> requests,
                                                         RequestTimePoint observed_at);

} // namespace helix::scheduler
