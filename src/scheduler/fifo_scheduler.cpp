#include "helix/scheduler/fifo_scheduler.hpp"

#include "helix/core/error.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>

namespace helix::scheduler {
namespace {

std::string scheduler_context(std::string_view operation) {
  return "scheduler::FifoScheduler::" + std::string(operation);
}

std::string request_error(RequestId id, std::string_view requirement) {
  std::ostringstream message;
  message << "request " << id.value() << ' ' << requirement;
  return message.str();
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs, std::string_view context) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw_error(ErrorCode::kInvalidArgument, std::string(context) + ": size addition overflow");
  }
  return lhs + rhs;
}

} // namespace

std::string_view work_kind_name(WorkKind kind) {
  switch (kind) {
  case WorkKind::kPrefill:
    return "prefill";
  case WorkKind::kDecode:
    return "decode";
  }
  std::ostringstream message;
  message << "scheduler::work_kind_name: unknown value " << static_cast<std::uint32_t>(kind);
  throw_error(ErrorCode::kInvalidArgument, message.str());
}

FifoScheduler::FifoScheduler(SchedulerConfig config) : config_(config) {
  HELIX_CHECK(config_.max_tokens_per_iteration != 0, ErrorCode::kInvalidArgument,
              "scheduler::FifoScheduler: max tokens per iteration must be positive");
  HELIX_CHECK(config_.max_running_requests != 0, ErrorCode::kInvalidArgument,
              "scheduler::FifoScheduler: max running requests must be positive");
  HELIX_CHECK(config_.kv_block_tokens != 0, ErrorCode::kInvalidArgument,
              "scheduler::FifoScheduler: KV block token count must be positive");
  running_.reserve(config_.max_running_requests);
}

void FifoScheduler::submit(Request request) {
  const RequestId id = request.id();
  HELIX_CHECK(id.valid(), ErrorCode::kInvalidArgument,
              "scheduler::FifoScheduler::submit: request has an invalid ID");
  HELIX_CHECK(request.state() == RequestState::kWaiting, ErrorCode::kInvalidArgument,
              scheduler_context("submit") + ": request must be waiting");
  HELIX_CHECK(!contains(id), ErrorCode::kInvalidArgument,
              scheduler_context("submit") + ": duplicate " + request_error(id, "ID"));
  HELIX_CHECK(cumulative_stats_.submitted_count != std::numeric_limits<std::uint64_t>::max(),
              ErrorCode::kResourceExhausted,
              scheduler_context("submit") + ": submitted counter exhausted");

  auto owned_request = std::make_unique<Request>(std::move(request));
  const auto [position, inserted] = requests_.emplace(id, std::move(owned_request));
  HELIX_ASSERT(inserted, "duplicate request passed prevalidated scheduler insertion");
  try {
    waiting_.push_back(id);
  } catch (...) {
    requests_.erase(position);
    throw;
  }
  ++cumulative_stats_.submitted_count;
  verify_invariants();
}

std::size_t FifoScheduler::current_kv_tokens(const Request& request) const {
  return checked_add(request.prompt_tokens_processed(), request.generated_tokens().size(),
                     scheduler_context("current_kv_tokens"));
}

std::size_t FifoScheduler::kv_block_count(std::size_t tokens) const noexcept {
  return tokens / config_.kv_block_tokens +
         static_cast<std::size_t>(tokens % config_.kv_block_tokens != 0);
}

std::size_t FifoScheduler::additional_kv_blocks(std::size_t current_tokens,
                                                std::size_t appended_tokens) const {
  const std::size_t resulting_tokens =
      checked_add(current_tokens, appended_tokens, scheduler_context("additional_kv_blocks"));
  return kv_block_count(resulting_tokens) - kv_block_count(current_tokens);
}

std::size_t FifoScheduler::max_append_for_blocks(std::size_t current_tokens,
                                                 std::size_t desired_tokens,
                                                 std::size_t available_blocks) const {
  std::size_t low = 0;
  std::size_t high = desired_tokens;
  while (low < high) {
    const std::size_t distance = high - low;
    const std::size_t midpoint = low + distance / 2 + distance % 2;
    if (additional_kv_blocks(current_tokens, midpoint) <= available_blocks) {
      low = midpoint;
    } else {
      high = midpoint - 1;
    }
  }
  return low;
}

void FifoScheduler::require_no_active_iteration(std::string_view operation) const {
  HELIX_CHECK(!active_iteration_.has_value(), ErrorCode::kInvalidArgument,
              scheduler_context(operation) + ": an iteration is already in flight");
}

void FifoScheduler::require_time_not_before_scheduler(RequestTimePoint now,
                                                      std::string_view operation) const {
  HELIX_CHECK(!latest_scheduler_event_.has_value() || now >= *latest_scheduler_event_,
              ErrorCode::kInvalidArgument,
              scheduler_context(operation) + ": event time precedes prior scheduler activity");
}

ScheduleResult FifoScheduler::schedule_iteration(std::size_t available_kv_blocks,
                                                 RequestTimePoint now) {
  constexpr std::string_view kOperation = "schedule_iteration";
  require_no_active_iteration(kOperation);
  require_time_not_before_scheduler(now, kOperation);

  IterationBatch planned;
  planned.scheduled_at = now;
  planned.work.reserve(config_.max_running_requests);
  std::vector<RequestId> admissions;
  admissions.reserve(config_.max_running_requests - running_.size());

  std::size_t remaining_tokens = config_.max_tokens_per_iteration;
  std::size_t remaining_blocks = available_kv_blocks;
  bool kv_limited = false;
  bool stop = false;

  auto plan_request = [&](const Request& request, bool waiting_request) {
    const RequestState state = waiting_request ? RequestState::kPrefill : request.state();
    HELIX_ASSERT(state == RequestState::kPrefill || state == RequestState::kDecoding,
                 "scheduler running set contains a request outside executable states");
    HELIX_CHECK(now >= request.created_at(), ErrorCode::kInvalidArgument,
                scheduler_context(kOperation) + ": schedule time precedes " +
                    request_error(request.id(), "creation"));

    const std::size_t desired_tokens =
        state == RequestState::kPrefill ? request.remaining_prompt_tokens() : 1;
    HELIX_ASSERT(desired_tokens != 0, "scheduler selected zero work for an active request");
    const std::size_t budgeted_tokens = std::min(desired_tokens, remaining_tokens);
    const std::size_t current_tokens = current_kv_tokens(request);
    const std::size_t scheduled_tokens =
        max_append_for_blocks(current_tokens, budgeted_tokens, remaining_blocks);
    if (scheduled_tokens == 0) {
      kv_limited = true;
      stop = true;
      return;
    }

    const std::size_t blocks_before = kv_block_count(current_tokens);
    const std::size_t new_blocks = additional_kv_blocks(current_tokens, scheduled_tokens);
    planned.work.push_back(ScheduledWork{
        request.id(), state == RequestState::kPrefill ? WorkKind::kPrefill : WorkKind::kDecode,
        state == RequestState::kPrefill ? request.prompt_tokens_processed() : current_tokens,
        scheduled_tokens, blocks_before, new_blocks});
    planned.scheduled_tokens += scheduled_tokens;
    planned.new_kv_blocks += new_blocks;
    remaining_tokens -= scheduled_tokens;
    remaining_blocks -= new_blocks;

    if (scheduled_tokens < budgeted_tokens) {
      kv_limited = true;
      stop = true;
    } else if (remaining_tokens == 0) {
      stop = true;
    }
  };

  for (const RequestId id : running_) {
    if (stop) {
      break;
    }
    plan_request(request(id), false);
  }

  std::size_t waiting_index = 0;
  while (!stop && running_.size() + admissions.size() < config_.max_running_requests &&
         waiting_index < waiting_.size()) {
    const RequestId id = waiting_[waiting_index];
    const Request& queued = request(id);
    const std::size_t work_count_before = planned.work.size();
    plan_request(queued, true);
    if (planned.work.size() != work_count_before) {
      admissions.push_back(id);
      ++waiting_index;
    }
  }

  ScheduleResult result;
  result.unused_token_budget = remaining_tokens;
  result.unused_kv_blocks = remaining_blocks;
  result.token_budget_exhausted = remaining_tokens == 0;
  result.kv_capacity_limited = kv_limited;
  if (planned.work.empty()) {
    return result;
  }

  HELIX_CHECK(!iteration_ids_exhausted_, ErrorCode::kResourceExhausted,
              scheduler_context(kOperation) + ": iteration ID space exhausted");
  HELIX_CHECK(cumulative_stats_.scheduled_iteration_count !=
                  std::numeric_limits<std::uint64_t>::max(),
              ErrorCode::kResourceExhausted,
              scheduler_context(kOperation) + ": scheduled-iteration counter exhausted");

  std::uint64_t prefill_tokens = 0;
  std::uint64_t decode_tokens = 0;
  for (const ScheduledWork& work : planned.work) {
    if (work.kind == WorkKind::kPrefill) {
      prefill_tokens =
          checked_add(prefill_tokens, work.token_count, scheduler_context("prefill_counter"));
    } else {
      decode_tokens =
          checked_add(decode_tokens, work.token_count, scheduler_context("decode_counter"));
    }
  }
  HELIX_CHECK(prefill_tokens <= std::numeric_limits<std::uint64_t>::max() -
                                    cumulative_stats_.scheduled_prefill_tokens,
              ErrorCode::kResourceExhausted,
              scheduler_context(kOperation) + ": prefill-token counter exhausted");
  HELIX_CHECK(decode_tokens <= std::numeric_limits<std::uint64_t>::max() -
                                   cumulative_stats_.scheduled_decode_tokens,
              ErrorCode::kResourceExhausted,
              scheduler_context(kOperation) + ": decode-token counter exhausted");

  planned.id = IterationId{next_iteration_id_};
  IterationBatch active_copy = planned;
  for (const RequestId id : admissions) {
    HELIX_ASSERT(!waiting_.empty() && waiting_.front() == id,
                 "scheduler admission plan diverged from the FIFO queue");
    mutable_request(id).begin_prefill(now);
    waiting_.pop_front();
    running_.push_back(id);
  }

  if (next_iteration_id_ == std::numeric_limits<std::uint64_t>::max()) {
    iteration_ids_exhausted_ = true;
  } else {
    ++next_iteration_id_;
  }
  active_iteration_ = ActiveIteration{std::move(active_copy)};
  latest_scheduler_event_ = now;
  ++cumulative_stats_.scheduled_iteration_count;
  cumulative_stats_.scheduled_prefill_tokens += prefill_tokens;
  cumulative_stats_.scheduled_decode_tokens += decode_tokens;
  result.batch = std::move(planned);
  verify_invariants();
  return result;
}

CompletionReport FifoScheduler::complete_iteration(IterationId iteration_id,
                                                   std::span<const WorkCompletion> completions,
                                                   RequestTimePoint now) {
  constexpr std::string_view kOperation = "complete_iteration";
  HELIX_CHECK(active_iteration_.has_value(), ErrorCode::kInvalidArgument,
              scheduler_context(kOperation) + ": no iteration is in flight");
  HELIX_CHECK(iteration_id.valid() && iteration_id == active_iteration_->batch.id,
              ErrorCode::kInvalidArgument,
              scheduler_context(kOperation) + ": iteration ID does not match the active batch");
  require_time_not_before_scheduler(now, kOperation);
  HELIX_CHECK(now >= active_iteration_->batch.scheduled_at, ErrorCode::kInvalidArgument,
              scheduler_context(kOperation) + ": completion precedes batch scheduling");
  HELIX_CHECK(completions.size() == active_iteration_->batch.work.size(),
              ErrorCode::kInvalidArgument,
              scheduler_context(kOperation) + ": completion count does not match work count");
  HELIX_CHECK(cumulative_stats_.completed_iteration_count !=
                  std::numeric_limits<std::uint64_t>::max(),
              ErrorCode::kResourceExhausted,
              scheduler_context(kOperation) + ": completed-iteration counter exhausted");

  CompletionReport report;
  report.iteration_id = iteration_id;
  report.retired_requests.reserve(completions.size());

  for (std::size_t index = 0; index < completions.size(); ++index) {
    const ScheduledWork& work = active_iteration_->batch.work[index];
    const WorkCompletion& completion = completions[index];
    HELIX_CHECK(completion.request_id == work.request_id, ErrorCode::kInvalidArgument,
                scheduler_context(kOperation) + ": completion order/request ID mismatch");
    HELIX_CHECK(completion.status == WorkStatus::kSucceeded ||
                    completion.status == WorkStatus::kFailed,
                ErrorCode::kInvalidArgument,
                scheduler_context(kOperation) + ": completion has an unknown status");
    Request& pending = mutable_request(work.request_id);
    HELIX_CHECK(!pending.terminal(), ErrorCode::kInternal,
                scheduler_context(kOperation) + ": active work references a terminal request");
    if (work.kind == WorkKind::kPrefill) {
      HELIX_CHECK(pending.state() == RequestState::kPrefill &&
                      pending.prompt_tokens_processed() == work.token_offset &&
                      work.token_count <= pending.remaining_prompt_tokens(),
                  ErrorCode::kInternal,
                  scheduler_context(kOperation) + ": prefill work no longer matches request state");
    } else {
      HELIX_CHECK(pending.state() == RequestState::kDecoding && work.token_count == 1 &&
                      current_kv_tokens(pending) == work.token_offset &&
                      pending.remaining_generation_tokens() != 0,
                  ErrorCode::kInternal,
                  scheduler_context(kOperation) + ": decode work no longer matches request state");
    }

    if (completion.status == WorkStatus::kFailed) {
      HELIX_CHECK(!completion.error_context.empty() && !completion.generated_token.has_value() &&
                      !completion.end_of_sequence,
                  ErrorCode::kInvalidArgument,
                  scheduler_context(kOperation) +
                      ": failed work requires only a nonempty error context");
      report.retired_requests.push_back(
          RetiredRequest{work.request_id, RequestState::kFailed, work.kv_blocks_after()});
    } else if (work.kind == WorkKind::kPrefill) {
      HELIX_CHECK(completion.error_context.empty() && !completion.generated_token.has_value() &&
                      !completion.end_of_sequence,
                  ErrorCode::kInvalidArgument,
                  scheduler_context(kOperation) +
                      ": successful prefill cannot carry a token, EOS, or error");
      if (work.token_count == pending.remaining_prompt_tokens() && pending.max_new_tokens() == 0) {
        report.retired_requests.push_back(
            RetiredRequest{work.request_id, RequestState::kFinished, work.kv_blocks_after()});
      }
    } else {
      HELIX_CHECK(completion.error_context.empty() && completion.generated_token.has_value(),
                  ErrorCode::kInvalidArgument,
                  scheduler_context(kOperation) +
                      ": successful decode requires one token and no error");
      if (completion.end_of_sequence || pending.remaining_generation_tokens() == 1) {
        report.retired_requests.push_back(
            RetiredRequest{work.request_id, RequestState::kFinished, work.kv_blocks_after()});
      }
    }
  }

  for (const RetiredRequest& retired : report.retired_requests) {
    report.released_kv_blocks =
        checked_add(report.released_kv_blocks, retired.released_kv_blocks,
                    scheduler_context("complete_iteration release accounting"));
  }

  // Finish every potentially throwing allocation before changing logical state.
  // Capacity growth is not externally observable progress, and it makes each
  // subsequent GeneratedToken push nonallocating.
  std::vector<std::string> staged_failure_contexts(completions.size());
  for (std::size_t index = 0; index < completions.size(); ++index) {
    const ScheduledWork& work = active_iteration_->batch.work[index];
    const WorkCompletion& completion = completions[index];
    if (completion.status == WorkStatus::kFailed) {
      staged_failure_contexts[index] = completion.error_context;
    } else if (work.kind == WorkKind::kDecode) {
      Request& pending = mutable_request(work.request_id);
      pending.reserve_generated_capacity(pending.generated_tokens().size() + 1);
    }
  }

  for (std::size_t index = 0; index < completions.size(); ++index) {
    const ScheduledWork& work = active_iteration_->batch.work[index];
    const WorkCompletion& completion = completions[index];
    Request& pending = mutable_request(work.request_id);
    if (completion.status == WorkStatus::kFailed) {
      pending.fail(now, std::move(staged_failure_contexts[index]));
    } else if (work.kind == WorkKind::kPrefill) {
      pending.record_prefill_progress(work.token_count, now);
    } else {
      pending.append_generated_token(*completion.generated_token, now);
      if (completion.end_of_sequence && !pending.terminal()) {
        pending.finish_decoding(now);
      }
    }
  }

  for (const RetiredRequest& retired : report.retired_requests) {
    HELIX_ASSERT(mutable_request(retired.request_id).state() == retired.state,
                 "scheduler retirement prediction diverged from request transition");
    remove_running(retired.request_id);
  }

  active_iteration_.reset();
  latest_scheduler_event_ = now;
  ++cumulative_stats_.completed_iteration_count;
  verify_invariants();
  return report;
}

void FifoScheduler::remove_waiting(RequestId request_id) {
  const auto position = std::find(waiting_.begin(), waiting_.end(), request_id);
  HELIX_ASSERT(position != waiting_.end(), "scheduler waiting request is absent from FIFO queue");
  waiting_.erase(position);
}

void FifoScheduler::remove_running(RequestId request_id) {
  const auto position = std::find(running_.begin(), running_.end(), request_id);
  HELIX_ASSERT(position != running_.end(), "scheduler running request is absent from active order");
  running_.erase(position);
}

RetiredRequest FifoScheduler::terminate(RequestId request_id, RequestState terminal_state,
                                        RequestTimePoint now, std::string context) {
  require_no_active_iteration("terminate");
  require_time_not_before_scheduler(now, "terminate");
  Request& target = mutable_request(request_id);
  HELIX_CHECK(!target.terminal(), ErrorCode::kInvalidArgument,
              scheduler_context("terminate") + ": " + request_error(request_id, "is terminal"));
  const std::size_t blocks = kv_block_count(current_kv_tokens(target));
  const RequestState previous_state = target.state();
  if (terminal_state == RequestState::kCancelled) {
    target.cancel(now, std::move(context));
  } else {
    HELIX_ASSERT(terminal_state == RequestState::kFailed,
                 "scheduler terminate received a nonterminal target state");
    target.fail(now, std::move(context));
  }
  if (previous_state == RequestState::kWaiting) {
    remove_waiting(request_id);
  } else {
    remove_running(request_id);
  }
  latest_scheduler_event_ = now;
  verify_invariants();
  return RetiredRequest{request_id, terminal_state, blocks};
}

RetiredRequest FifoScheduler::cancel(RequestId request_id, RequestTimePoint now,
                                     std::string context) {
  return terminate(request_id, RequestState::kCancelled, now, std::move(context));
}

RetiredRequest FifoScheduler::fail(RequestId request_id, RequestTimePoint now,
                                   std::string context) {
  return terminate(request_id, RequestState::kFailed, now, std::move(context));
}

bool FifoScheduler::contains(RequestId request_id) const noexcept {
  return requests_.contains(request_id);
}

Request& FifoScheduler::mutable_request(RequestId request_id) {
  const auto position = requests_.find(request_id);
  HELIX_CHECK(position != requests_.end(), ErrorCode::kOutOfRange,
              scheduler_context("request") + ": " + request_error(request_id, "was not found"));
  return *position->second;
}

const Request& FifoScheduler::request(RequestId request_id) const {
  const auto position = requests_.find(request_id);
  HELIX_CHECK(position != requests_.end(), ErrorCode::kOutOfRange,
              scheduler_context("request") + ": " + request_error(request_id, "was not found"));
  return *position->second;
}

std::span<const RequestId> FifoScheduler::running_requests() const noexcept {
  return running_;
}

std::vector<RequestId> FifoScheduler::waiting_requests() const {
  return {waiting_.begin(), waiting_.end()};
}

bool FifoScheduler::iteration_in_flight() const noexcept {
  return active_iteration_.has_value();
}

SchedulerStats FifoScheduler::stats() const {
  SchedulerStats result = cumulative_stats_;
  result.request_count = requests_.size();
  result.waiting_count = waiting_.size();
  result.running_count = running_.size();
  for (const auto& [id, owned] : requests_) {
    static_cast<void>(id);
    switch (owned->state()) {
    case RequestState::kFinished:
      ++result.finished_count;
      break;
    case RequestState::kCancelled:
      ++result.cancelled_count;
      break;
    case RequestState::kFailed:
      ++result.failed_count;
      break;
    case RequestState::kWaiting:
    case RequestState::kPrefill:
    case RequestState::kDecoding:
      break;
    }
  }
  return result;
}

SchedulerMetrics FifoScheduler::metrics(RequestTimePoint observed_at) const {
  require_time_not_before_scheduler(observed_at, "metrics");
  std::vector<const Request*> requests;
  requests.reserve(requests_.size());
  for (const auto& [id, owned] : requests_) {
    static_cast<void>(id);
    requests.push_back(owned.get());
  }
  return summarize_request_metrics(requests, observed_at);
}

const SchedulerConfig& FifoScheduler::config() const noexcept {
  return config_;
}

void FifoScheduler::verify_invariants() const {
#ifndef NDEBUG
  HELIX_ASSERT(running_.size() <= config_.max_running_requests,
               "scheduler exceeds configured running-request capacity");
  std::size_t categorized = 0;
  for (const auto& [id, owned] : requests_) {
    std::size_t waiting_occurrences = 0;
    for (const RequestId queued : waiting_) {
      waiting_occurrences += static_cast<std::size_t>(queued == id);
    }
    std::size_t running_occurrences = 0;
    for (const RequestId active : running_) {
      running_occurrences += static_cast<std::size_t>(active == id);
    }
    HELIX_ASSERT(waiting_occurrences <= 1 && running_occurrences <= 1,
                 "scheduler contains a duplicate queue/order entry");
    switch (owned->state()) {
    case RequestState::kWaiting:
      HELIX_ASSERT(waiting_occurrences == 1 && running_occurrences == 0,
                   "waiting request membership is inconsistent");
      break;
    case RequestState::kPrefill:
    case RequestState::kDecoding:
      HELIX_ASSERT(waiting_occurrences == 0 && running_occurrences == 1,
                   "running request membership is inconsistent");
      break;
    case RequestState::kFinished:
    case RequestState::kCancelled:
    case RequestState::kFailed:
      HELIX_ASSERT(waiting_occurrences == 0 && running_occurrences == 0,
                   "terminal request remains schedulable");
      break;
    }
    ++categorized;
  }
  HELIX_ASSERT(categorized == requests_.size(), "scheduler request accounting is inconsistent");
  if (active_iteration_.has_value()) {
    HELIX_ASSERT(active_iteration_->batch.id.valid() && !active_iteration_->batch.work.empty(),
                 "scheduler active iteration metadata is invalid");
  }
#endif
}

} // namespace helix::scheduler
