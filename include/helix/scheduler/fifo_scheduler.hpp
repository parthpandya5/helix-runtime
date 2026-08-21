#pragma once

#include "helix/scheduler/request.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace helix::scheduler {

class IterationId final {
public:
  IterationId() = default;
  explicit IterationId(std::uint64_t value) noexcept : value_(value) {}

  [[nodiscard]] bool valid() const noexcept {
    return value_ != 0;
  }
  [[nodiscard]] std::uint64_t value() const noexcept {
    return value_;
  }

  auto operator<=>(const IterationId&) const = default;

private:
  std::uint64_t value_ = 0;
};

enum class WorkKind : std::uint8_t {
  kPrefill,
  kDecode,
};

enum class WorkStatus : std::uint8_t {
  kSucceeded,
  kFailed,
};

[[nodiscard]] std::string_view work_kind_name(WorkKind kind);

struct ScheduledWork {
  RequestId request_id;
  WorkKind kind = WorkKind::kPrefill;
  std::size_t token_offset = 0;
  std::size_t token_count = 0;
  std::size_t kv_blocks_before = 0;
  std::size_t new_kv_blocks = 0;

  [[nodiscard]] std::size_t kv_blocks_after() const noexcept {
    return kv_blocks_before + new_kv_blocks;
  }
};

struct IterationBatch {
  IterationId id;
  RequestTimePoint scheduled_at;
  std::vector<ScheduledWork> work;
  std::size_t scheduled_tokens = 0;
  std::size_t new_kv_blocks = 0;
};

struct ScheduleResult {
  std::optional<IterationBatch> batch;
  std::size_t unused_token_budget = 0;
  std::size_t unused_kv_blocks = 0;
  bool token_budget_exhausted = false;
  bool kv_capacity_limited = false;
};

struct WorkCompletion {
  RequestId request_id;
  WorkStatus status = WorkStatus::kSucceeded;
  std::optional<TokenId> generated_token;
  bool end_of_sequence = false;
  std::string error_context;
};

struct RetiredRequest {
  RequestId request_id;
  RequestState state = RequestState::kFinished;
  std::size_t released_kv_blocks = 0;
};

struct CompletionReport {
  IterationId iteration_id;
  std::vector<RetiredRequest> retired_requests;
  std::size_t released_kv_blocks = 0;
};

struct SchedulerConfig {
  std::size_t max_tokens_per_iteration = 0;
  std::size_t max_running_requests = 0;
  std::size_t kv_block_tokens = 0;
};

struct SchedulerStats {
  std::size_t request_count = 0;
  std::size_t waiting_count = 0;
  std::size_t running_count = 0;
  std::size_t finished_count = 0;
  std::size_t cancelled_count = 0;
  std::size_t failed_count = 0;
  std::uint64_t submitted_count = 0;
  std::uint64_t scheduled_iteration_count = 0;
  std::uint64_t completed_iteration_count = 0;
  std::uint64_t scheduled_prefill_tokens = 0;
  std::uint64_t scheduled_decode_tokens = 0;
};

// FifoScheduler is single-thread owned. It owns every Request and never owns KV
// memory; each schedule call consumes a free-block snapshot and reports the block
// reservations/releases that the runtime's KV authority must apply.
class FifoScheduler final {
public:
  explicit FifoScheduler(SchedulerConfig config);

  FifoScheduler(const FifoScheduler&) = delete;
  FifoScheduler& operator=(const FifoScheduler&) = delete;
  FifoScheduler(FifoScheduler&&) = delete;
  FifoScheduler& operator=(FifoScheduler&&) = delete;

  void submit(Request request);
  [[nodiscard]] ScheduleResult schedule_iteration(std::size_t available_kv_blocks,
                                                  RequestTimePoint now);
  [[nodiscard]] CompletionReport complete_iteration(IterationId iteration_id,
                                                    std::span<const WorkCompletion> completions,
                                                    RequestTimePoint now);
  [[nodiscard]] RetiredRequest cancel(RequestId request_id, RequestTimePoint now,
                                      std::string context = "cancelled by caller");
  [[nodiscard]] RetiredRequest fail(RequestId request_id, RequestTimePoint now,
                                    std::string context);

  [[nodiscard]] bool contains(RequestId request_id) const noexcept;
  [[nodiscard]] const Request& request(RequestId request_id) const;
  [[nodiscard]] std::span<const RequestId> running_requests() const noexcept;
  [[nodiscard]] std::vector<RequestId> waiting_requests() const;
  [[nodiscard]] bool iteration_in_flight() const noexcept;
  [[nodiscard]] SchedulerStats stats() const;
  [[nodiscard]] const SchedulerConfig& config() const noexcept;

private:
  struct ActiveIteration {
    IterationBatch batch;
  };

  [[nodiscard]] Request& mutable_request(RequestId request_id);
  [[nodiscard]] std::size_t current_kv_tokens(const Request& request) const;
  [[nodiscard]] std::size_t kv_block_count(std::size_t tokens) const noexcept;
  [[nodiscard]] std::size_t additional_kv_blocks(std::size_t current_tokens,
                                                 std::size_t appended_tokens) const;
  [[nodiscard]] std::size_t max_append_for_blocks(std::size_t current_tokens,
                                                  std::size_t desired_tokens,
                                                  std::size_t available_blocks) const;
  [[nodiscard]] RetiredRequest terminate(RequestId request_id, RequestState terminal_state,
                                         RequestTimePoint now, std::string context);
  void require_no_active_iteration(std::string_view operation) const;
  void require_time_not_before_scheduler(RequestTimePoint now, std::string_view operation) const;
  void remove_waiting(RequestId request_id);
  void remove_running(RequestId request_id);
  void verify_invariants() const;

  SchedulerConfig config_;
  std::map<RequestId, std::unique_ptr<Request>> requests_;
  std::deque<RequestId> waiting_;
  std::vector<RequestId> running_;
  std::optional<ActiveIteration> active_iteration_;
  std::optional<RequestTimePoint> latest_scheduler_event_;
  std::uint64_t next_iteration_id_ = 1;
  bool iteration_ids_exhausted_ = false;
  SchedulerStats cumulative_stats_;
};

} // namespace helix::scheduler
