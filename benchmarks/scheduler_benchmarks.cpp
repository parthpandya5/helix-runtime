#include "helix/scheduler/fifo_scheduler.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace helix::scheduler {
namespace {

std::vector<TokenId> make_prompt(std::size_t size, std::uint32_t seed) {
  std::vector<TokenId> tokens;
  tokens.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    tokens.emplace_back(seed + static_cast<std::uint32_t>(index));
  }
  return tokens;
}

void BM_FifoSchedulerTrace(benchmark::State& state) {
  const auto request_count = static_cast<std::size_t>(state.range(0));
  const auto prompt_tokens = static_cast<std::size_t>(state.range(1));
  const auto output_tokens = static_cast<std::size_t>(state.range(2));
  const auto max_running = static_cast<std::size_t>(state.range(3));
  const auto token_budget = static_cast<std::size_t>(state.range(4));
  const auto block_tokens = static_cast<std::size_t>(state.range(5));
  const std::size_t blocks_per_request =
      (prompt_tokens + output_tokens) / block_tokens +
      static_cast<std::size_t>((prompt_tokens + output_tokens) % block_tokens != 0);
  const std::size_t total_blocks = request_count * blocks_per_request;
  const std::size_t tokens_per_trace = request_count * (prompt_tokens + output_tokens);
  std::uint64_t total_scheduler_iterations = 0;

  for ([[maybe_unused]] auto iteration : state) {
    state.PauseTiming();
    auto scheduler =
        std::make_unique<FifoScheduler>(SchedulerConfig{token_budget, max_running, block_tokens});
    constexpr RequestTimePoint kCreated{};
    for (std::size_t request_index = 0; request_index < request_count; ++request_index) {
      scheduler->submit(Request(
          RequestId{request_index + 1},
          make_prompt(prompt_tokens, static_cast<std::uint32_t>(10'000 + request_index * 257)),
          output_tokens, kCreated));
    }
    std::size_t available_blocks = total_blocks;
    RequestTimePoint now = kCreated;
    state.ResumeTiming();

    std::uint64_t trace_iterations = 0;
    std::size_t finished_requests = 0;
    while (finished_requests != request_count) {
      const ScheduleResult scheduled = scheduler->schedule_iteration(available_blocks, now);
      if (!scheduled.batch.has_value()) {
        state.SkipWithError("scheduler made no progress with sufficient synthetic KV capacity");
        break;
      }
      available_blocks -= scheduled.batch->new_kv_blocks;
      std::vector<WorkCompletion> completions;
      completions.reserve(scheduled.batch->work.size());
      for (const ScheduledWork& work : scheduled.batch->work) {
        if (work.kind == WorkKind::kPrefill) {
          completions.push_back(
              WorkCompletion{work.request_id, WorkStatus::kSucceeded, std::nullopt, false, {}});
        } else {
          completions.push_back(
              WorkCompletion{work.request_id,
                             WorkStatus::kSucceeded,
                             TokenId{static_cast<std::uint32_t>(50'000 + work.request_id.value())},
                             false,
                             {}});
        }
      }
      now += std::chrono::microseconds(1);
      const CompletionReport completed =
          scheduler->complete_iteration(scheduled.batch->id, completions, now);
      available_blocks += completed.released_kv_blocks;
      for (const RetiredRequest& retired : completed.retired_requests) {
        finished_requests += static_cast<std::size_t>(retired.state == RequestState::kFinished);
      }
      now += std::chrono::microseconds(1);
      ++trace_iterations;
    }
    benchmark::DoNotOptimize(finished_requests);
    total_scheduler_iterations += trace_iterations;
  }

  state.counters["iterations_per_trace"] =
      static_cast<double>(total_scheduler_iterations) / static_cast<double>(state.iterations());
  state.counters["requests_per_trace"] = static_cast<double>(request_count);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(tokens_per_trace));
}

BENCHMARK(BM_FifoSchedulerTrace)
    ->Args({16, 8, 8, 8, 64, 16})
    ->Args({64, 32, 16, 16, 128, 16})
    ->Args({64, 128, 32, 32, 512, 16})
    ->Args({128, 17, 31, 32, 256, 7});

struct SyntheticRequestSpec {
  std::size_t prompt_tokens = 0;
  std::size_t output_tokens = 0;
};

struct SyntheticTraceResult {
  SchedulerMetrics metrics;
  std::uint64_t scheduler_iterations = 0;
};

std::vector<SyntheticRequestSpec>
make_mixed_workload(std::size_t request_count, std::size_t max_running, std::size_t prompt_tokens,
                    std::size_t long_output_tokens, std::size_t short_output_tokens) {
  std::vector<SyntheticRequestSpec> requests;
  requests.reserve(request_count);
  for (std::size_t index = 0; index < request_count; ++index) {
    const bool cohort_leader = index % max_running == 0;
    requests.push_back(SyntheticRequestSpec{prompt_tokens, cohort_leader ? long_output_tokens
                                                                         : short_output_tokens});
  }
  return requests;
}

std::size_t required_blocks(const SyntheticRequestSpec& request, std::size_t block_tokens) {
  const std::size_t tokens = request.prompt_tokens + request.output_tokens;
  return tokens / block_tokens + static_cast<std::size_t>(tokens % block_tokens != 0);
}

std::uint64_t run_until_finished(FifoScheduler& scheduler, std::size_t target_finished,
                                 std::size_t& available_blocks, RequestTimePoint& now) {
  std::uint64_t iterations = 0;
  std::size_t finished_requests = 0;
  while (finished_requests != target_finished) {
    const ScheduleResult scheduled = scheduler.schedule_iteration(available_blocks, now);
    if (!scheduled.batch.has_value()) {
      throw std::runtime_error("synthetic scheduler trace made no progress");
    }
    available_blocks -= scheduled.batch->new_kv_blocks;

    std::vector<WorkCompletion> completions;
    completions.reserve(scheduled.batch->work.size());
    for (const ScheduledWork& work : scheduled.batch->work) {
      if (work.kind == WorkKind::kPrefill) {
        completions.push_back(
            WorkCompletion{work.request_id, WorkStatus::kSucceeded, std::nullopt, false, {}});
      } else {
        completions.push_back(
            WorkCompletion{work.request_id,
                           WorkStatus::kSucceeded,
                           TokenId{static_cast<std::uint32_t>(100'000 + work.request_id.value())},
                           false,
                           {}});
      }
    }

    // One millisecond is a declared synthetic model-iteration cost. It drives
    // request timestamps only and is unrelated to this function's CPU duration.
    now += std::chrono::milliseconds(1);
    const CompletionReport completed =
        scheduler.complete_iteration(scheduled.batch->id, completions, now);
    available_blocks += completed.released_kv_blocks;
    for (const RetiredRequest& retired : completed.retired_requests) {
      finished_requests += static_cast<std::size_t>(retired.state == RequestState::kFinished);
    }
    ++iterations;
  }
  return iterations;
}

SyntheticTraceResult run_continuous_trace(const std::vector<SyntheticRequestSpec>& workload,
                                          std::size_t max_running, std::size_t token_budget,
                                          std::size_t block_tokens) {
  constexpr RequestTimePoint kCreated{};
  FifoScheduler scheduler(SchedulerConfig{token_budget, max_running, block_tokens});
  std::size_t available_blocks = 0;
  for (std::size_t index = 0; index < workload.size(); ++index) {
    const SyntheticRequestSpec& spec = workload[index];
    scheduler.submit(
        Request(RequestId{index + 1},
                make_prompt(spec.prompt_tokens, static_cast<std::uint32_t>(20'000 + index * 257)),
                spec.output_tokens, kCreated));
    available_blocks += required_blocks(spec, block_tokens);
  }

  RequestTimePoint now = kCreated;
  const std::uint64_t iterations =
      run_until_finished(scheduler, workload.size(), available_blocks, now);
  return SyntheticTraceResult{scheduler.metrics(now), iterations};
}

SyntheticTraceResult run_static_cohort_trace(const std::vector<SyntheticRequestSpec>& workload,
                                             std::size_t max_running, std::size_t token_budget,
                                             std::size_t block_tokens) {
  constexpr RequestTimePoint kCreated{};
  std::vector<std::unique_ptr<FifoScheduler>> cohorts;
  std::vector<const Request*> completed_requests;
  cohorts.reserve((workload.size() + max_running - 1) / max_running);
  completed_requests.reserve(workload.size());

  RequestTimePoint now = kCreated;
  std::uint64_t total_iterations = 0;
  for (std::size_t begin = 0; begin < workload.size(); begin += max_running) {
    const std::size_t end = std::min(begin + max_running, workload.size());
    auto scheduler =
        std::make_unique<FifoScheduler>(SchedulerConfig{token_budget, max_running, block_tokens});
    std::size_t available_blocks = 0;
    for (std::size_t index = begin; index < end; ++index) {
      const SyntheticRequestSpec& spec = workload[index];
      scheduler->submit(
          Request(RequestId{index + 1},
                  make_prompt(spec.prompt_tokens, static_cast<std::uint32_t>(30'000 + index * 257)),
                  spec.output_tokens, kCreated));
      available_blocks += required_blocks(spec, block_tokens);
    }
    total_iterations += run_until_finished(*scheduler, end - begin, available_blocks, now);
    for (std::size_t index = begin; index < end; ++index) {
      completed_requests.push_back(&scheduler->request(RequestId{index + 1}));
    }
    cohorts.push_back(std::move(scheduler));
  }

  return SyntheticTraceResult{summarize_request_metrics(completed_requests, now), total_iterations};
}

enum class SyntheticPolicy : std::uint8_t {
  kContinuous,
  kStaticCohort,
};

void BM_SyntheticBatchingPolicy(benchmark::State& state, SyntheticPolicy policy) {
  const auto request_count = static_cast<std::size_t>(state.range(0));
  const auto max_running = static_cast<std::size_t>(state.range(1));
  const auto prompt_tokens = static_cast<std::size_t>(state.range(2));
  const auto long_output_tokens = static_cast<std::size_t>(state.range(3));
  const auto short_output_tokens = static_cast<std::size_t>(state.range(4));
  const auto block_tokens = static_cast<std::size_t>(state.range(5));
  const std::size_t token_budget = max_running * prompt_tokens;
  const std::vector<SyntheticRequestSpec> workload = make_mixed_workload(
      request_count, max_running, prompt_tokens, long_output_tokens, short_output_tokens);

  SyntheticTraceResult result;
  for ([[maybe_unused]] auto iteration : state) {
    result = policy == SyntheticPolicy::kContinuous
                 ? run_continuous_trace(workload, max_running, token_budget, block_tokens)
                 : run_static_cohort_trace(workload, max_running, token_budget, block_tokens);
    benchmark::DoNotOptimize(result.scheduler_iterations);
    benchmark::DoNotOptimize(result.metrics.finished_count);
  }

  state.counters["modeled_iterations"] = static_cast<double>(result.scheduler_iterations);
  state.counters["modeled_makespan_ms"] = result.metrics.observation_seconds * 1'000.0;
  state.counters["modeled_requests_per_second"] = result.metrics.finished_requests_per_second;
  state.counters["modeled_tokens_per_second"] = result.metrics.generated_tokens_per_second;
  state.counters["modeled_inter_token_mean_ms"] =
      result.metrics.inter_token_latency.mean_seconds * 1'000.0;
  state.counters["modeled_latency_p50_ms"] = result.metrics.request_latency.p50_seconds * 1'000.0;
  state.counters["modeled_latency_p95_ms"] = result.metrics.request_latency.p95_seconds * 1'000.0;
  state.counters["modeled_ttft_p50_ms"] = result.metrics.time_to_first_token.p50_seconds * 1'000.0;
  state.counters["modeled_ttft_p95_ms"] = result.metrics.time_to_first_token.p95_seconds * 1'000.0;
  state.counters["modeled_wait_p50_ms"] = result.metrics.waiting_time.p50_seconds * 1'000.0;
  state.counters["modeled_wait_p95_ms"] = result.metrics.waiting_time.p95_seconds * 1'000.0;
  state.counters["modeled_wait_mean_ms"] = result.metrics.waiting_time.mean_seconds * 1'000.0;
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(request_count));
}

void BM_ContinuousBatchingPolicy(benchmark::State& state) {
  BM_SyntheticBatchingPolicy(state, SyntheticPolicy::kContinuous);
}

void BM_StaticCohortPolicy(benchmark::State& state) {
  BM_SyntheticBatchingPolicy(state, SyntheticPolicy::kStaticCohort);
}

BENCHMARK(BM_ContinuousBatchingPolicy)->Args({32, 8, 8, 32, 4, 16})->Args({128, 32, 17, 64, 8, 7});
BENCHMARK(BM_StaticCohortPolicy)->Args({32, 8, 8, 32, 4, 16})->Args({128, 32, 17, 64, 8, 7});

} // namespace
} // namespace helix::scheduler
