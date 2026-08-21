#include "helix/scheduler/fifo_scheduler.hpp"

#include <benchmark/benchmark.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
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
    while (scheduler->stats().finished_count != request_count) {
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
      now += std::chrono::microseconds(1);
      ++trace_iterations;
    }
    benchmark::DoNotOptimize(scheduler->stats().finished_count);
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

} // namespace
} // namespace helix::scheduler
