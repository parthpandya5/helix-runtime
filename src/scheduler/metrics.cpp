#include "helix/scheduler/metrics.hpp"

#include "helix/core/error.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace helix::scheduler {
namespace {

double seconds(RequestDuration duration) {
  return std::chrono::duration<double>(duration).count();
}

double interpolated_percentile(const std::vector<double>& sorted_samples, double probability) {
  HELIX_ASSERT(!sorted_samples.empty(), "percentile requires at least one sample");
  const double position = probability * static_cast<double>(sorted_samples.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return sorted_samples[lower] + (sorted_samples[upper] - sorted_samples[lower]) * fraction;
}

DurationStatistics summarize_durations(std::vector<double> samples) {
  DurationStatistics result;
  result.sample_count = samples.size();
  if (samples.empty()) {
    return result;
  }

  std::sort(samples.begin(), samples.end());
  result.minimum_seconds = samples.front();
  result.maximum_seconds = samples.back();
  result.mean_seconds =
      std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
  result.p50_seconds = interpolated_percentile(samples, 0.50);
  result.p95_seconds = interpolated_percentile(samples, 0.95);
  return result;
}

void checked_accumulate(std::uint64_t& destination, std::size_t value,
                        std::string_view counter_name) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  const auto addend = static_cast<std::uint64_t>(value);
  HELIX_CHECK(addend <= std::numeric_limits<std::uint64_t>::max() - destination,
              ErrorCode::kResourceExhausted,
              "scheduler::summarize_request_metrics: " + std::string(counter_name) +
                  " counter overflow");
  destination += addend;
}

} // namespace

SchedulerMetrics summarize_request_metrics(std::span<const Request* const> requests,
                                           RequestTimePoint observed_at) {
  SchedulerMetrics result;
  result.request_count = requests.size();
  if (requests.empty()) {
    return result;
  }

  std::set<RequestId> seen_ids;
  std::vector<double> waiting_samples;
  std::vector<double> ttft_samples;
  std::vector<double> latency_samples;
  std::vector<double> inter_token_samples;
  waiting_samples.reserve(requests.size());
  ttft_samples.reserve(requests.size());
  latency_samples.reserve(requests.size());

  RequestTimePoint earliest_created = observed_at;
  for (const Request* request : requests) {
    HELIX_CHECK(request != nullptr, ErrorCode::kInvalidArgument,
                "scheduler::summarize_request_metrics: request pointer must not be null");
    HELIX_CHECK(request->id().valid(), ErrorCode::kInvalidArgument,
                "scheduler::summarize_request_metrics: request has an invalid ID");
    HELIX_CHECK(seen_ids.insert(request->id()).second, ErrorCode::kInvalidArgument,
                "scheduler::summarize_request_metrics: duplicate request ID");
    HELIX_CHECK(observed_at >= request->latest_event_at(), ErrorCode::kInvalidArgument,
                "scheduler::summarize_request_metrics: observation precedes request activity");
    earliest_created = std::min(earliest_created, request->created_at());

    switch (request->state()) {
    case RequestState::kWaiting:
      ++result.waiting_count;
      break;
    case RequestState::kPrefill:
      ++result.prefill_count;
      break;
    case RequestState::kDecoding:
      ++result.decoding_count;
      break;
    case RequestState::kFinished:
      ++result.finished_count;
      break;
    case RequestState::kCancelled:
      ++result.cancelled_count;
      break;
    case RequestState::kFailed:
      ++result.failed_count;
      break;
    }

    checked_accumulate(result.prompt_tokens_processed, request->prompt_tokens_processed(),
                       "prompt-token");
    checked_accumulate(result.generated_tokens, request->generated_tokens().size(),
                       "generated-token");

    if (const auto waiting = request->waiting_time(); waiting.has_value()) {
      waiting_samples.push_back(seconds(*waiting));
    }
    if (const auto ttft = request->time_to_first_token(); ttft.has_value()) {
      ttft_samples.push_back(seconds(*ttft));
    }
    if (request->state() == RequestState::kFinished) {
      HELIX_ASSERT(request->completed_at().has_value(),
                   "finished request lacks a completion timestamp");
      latency_samples.push_back(seconds(*request->completed_at() - request->created_at()));
    }

    const std::span<const GeneratedToken> generated = request->generated_tokens();
    for (std::size_t index = 1; index < generated.size(); ++index) {
      HELIX_ASSERT(generated[index].emitted_at >= generated[index - 1].emitted_at,
                   "request token timestamps are not monotonic");
      inter_token_samples.push_back(
          seconds(generated[index].emitted_at - generated[index - 1].emitted_at));
    }
  }

  result.observation_seconds = seconds(observed_at - earliest_created);
  HELIX_ASSERT(result.observation_seconds >= 0.0, "metrics observation window is negative");
  HELIX_CHECK(result.generated_tokens <=
                  std::numeric_limits<std::uint64_t>::max() - result.prompt_tokens_processed,
              ErrorCode::kResourceExhausted,
              "scheduler::summarize_request_metrics: processed-token counter overflow");
  const std::uint64_t processed = result.prompt_tokens_processed + result.generated_tokens;
  if (result.observation_seconds > 0.0) {
    result.throughput_defined = true;
    result.finished_requests_per_second =
        static_cast<double>(result.finished_count) / result.observation_seconds;
    result.generated_tokens_per_second =
        static_cast<double>(result.generated_tokens) / result.observation_seconds;
    result.processed_tokens_per_second =
        static_cast<double>(processed) / result.observation_seconds;
  }

  result.waiting_time = summarize_durations(std::move(waiting_samples));
  result.time_to_first_token = summarize_durations(std::move(ttft_samples));
  result.request_latency = summarize_durations(std::move(latency_samples));
  result.inter_token_latency = summarize_durations(std::move(inter_token_samples));
  return result;
}

} // namespace helix::scheduler
