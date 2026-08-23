#pragma once

#include "helix/kv_cache/prefix.hpp"
#include "helix/model/llama.hpp"
#include "helix/scheduler/fifo_scheduler.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace helix::runtime {

enum class CpuPrefillMode : std::uint8_t {
  kSequentialDecode,
  kFullPrompt,
};

struct CpuRuntimeConfig {
  CpuRuntimeConfig() = default;
  CpuRuntimeConfig(
      scheduler::SchedulerConfig scheduler_config, std::size_t kv_blocks,
      CpuPrefillMode selected_prefill_mode = CpuPrefillMode::kFullPrompt,
      model::PagedDecodeMode selected_decode_mode = model::PagedDecodeMode::kDirectBlockTable,
      std::optional<kv_cache::PagedPrefixCacheConfig> selected_prefix_cache = std::nullopt)
      : scheduler(scheduler_config), physical_kv_blocks(kv_blocks),
        prefill_mode(selected_prefill_mode), paged_decode_mode(selected_decode_mode),
        prefix_cache(selected_prefix_cache) {}

  scheduler::SchedulerConfig scheduler;
  std::size_t physical_kv_blocks = 0;
  CpuPrefillMode prefill_mode = CpuPrefillMode::kFullPrompt;
  model::PagedDecodeMode paged_decode_mode = model::PagedDecodeMode::kDirectBlockTable;
  std::optional<kv_cache::PagedPrefixCacheConfig> prefix_cache;
};

struct CpuRuntimeIterationResult {
  std::optional<scheduler::IterationId> iteration_id;
  std::vector<scheduler::ScheduledWork> work;
  std::vector<scheduler::WorkCompletion> completions;
  std::vector<scheduler::RetiredRequest> retired_requests;
  std::size_t free_kv_blocks_before = 0;
  std::size_t free_kv_blocks_after = 0;
  std::size_t scheduled_prefill_tokens = 0;
  std::size_t scheduled_decode_tokens = 0;
  bool token_budget_exhausted = false;
  bool kv_capacity_limited = false;
  std::size_t prefix_entries_evicted = 0;
};

struct CpuRuntimeStats {
  scheduler::SchedulerStats scheduler;
  kv_cache::PagedKvCacheStats kv_cache;
  std::size_t active_session_count = 0;
  std::uint64_t executed_iteration_count = 0;
  std::uint64_t failed_work_item_count = 0;
  std::uint64_t full_prompt_prefill_count = 0;
  std::uint64_t sequential_prefill_token_count = 0;
  std::optional<kv_cache::PagedPrefixCacheStats> prefix_cache;
  std::uint64_t prefix_publication_failure_count = 0;
};

// Synchronous CPU correctness engine. One thread owns the scheduler, model, and
// global paged KV pool. A call to run_iteration schedules, executes, completes,
// and reclaims one whole batch before returning; no scheduler iteration escapes
// in flight. CUDA streams and HelixAllocator-backed Tensor storage are not part of
// this CPU baseline.
class CpuRuntimeEngine final {
public:
  CpuRuntimeEngine(model::CpuLlamaModel model, CpuRuntimeConfig config);
  ~CpuRuntimeEngine();

  CpuRuntimeEngine(const CpuRuntimeEngine&) = delete;
  CpuRuntimeEngine& operator=(const CpuRuntimeEngine&) = delete;
  CpuRuntimeEngine(CpuRuntimeEngine&&) = delete;
  CpuRuntimeEngine& operator=(CpuRuntimeEngine&&) = delete;

  void submit(scheduler::Request request,
              std::optional<scheduler::TokenId> end_of_sequence = std::nullopt);
  [[nodiscard]] CpuRuntimeIterationResult run_iteration(scheduler::RequestTimePoint now);
  [[nodiscard]] scheduler::RetiredRequest cancel(scheduler::RequestId request_id,
                                                 scheduler::RequestTimePoint now,
                                                 std::string context = "cancelled by caller");

  [[nodiscard]] bool contains(scheduler::RequestId request_id) const noexcept;
  [[nodiscard]] bool has_live_requests() const noexcept;
  [[nodiscard]] const scheduler::Request& request(scheduler::RequestId request_id) const;
  [[nodiscard]] scheduler::SchedulerMetrics metrics(scheduler::RequestTimePoint observed_at) const;
  [[nodiscard]] CpuRuntimeStats stats() const;
  [[nodiscard]] const CpuRuntimeConfig& config() const noexcept;
  [[nodiscard]] const model::LlamaConfig& model_config() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace helix::runtime
