#include "helix/scheduler/request.hpp"

#include "helix/core/error.hpp"

#include <sstream>
#include <utility>

namespace helix::scheduler {
namespace {

[[noreturn]] void throw_invalid_enum(std::string_view type, std::uint32_t value) {
  std::ostringstream message;
  message << "scheduler::" << type << ": unknown value " << value;
  throw_error(ErrorCode::kInvalidArgument, message.str());
}

std::string request_context(RequestId id, std::string_view operation) {
  std::ostringstream message;
  message << "scheduler::Request(" << id.value() << ")::" << operation;
  return message.str();
}

} // namespace

std::string_view request_state_name(RequestState state) {
  switch (state) {
  case RequestState::kWaiting:
    return "waiting";
  case RequestState::kPrefill:
    return "prefill";
  case RequestState::kDecoding:
    return "decoding";
  case RequestState::kFinished:
    return "finished";
  case RequestState::kCancelled:
    return "cancelled";
  case RequestState::kFailed:
    return "failed";
  }
  throw_invalid_enum("request_state_name", static_cast<std::uint32_t>(state));
}

std::string_view finish_reason_name(FinishReason reason) {
  switch (reason) {
  case FinishReason::kNone:
    return "none";
  case FinishReason::kLength:
    return "length";
  case FinishReason::kEndOfSequence:
    return "end_of_sequence";
  case FinishReason::kCancelled:
    return "cancelled";
  case FinishReason::kFailed:
    return "failed";
  }
  throw_invalid_enum("finish_reason_name", static_cast<std::uint32_t>(reason));
}

bool request_state_is_terminal(RequestState state) noexcept {
  return state == RequestState::kFinished || state == RequestState::kCancelled ||
         state == RequestState::kFailed;
}

bool request_state_transition_allowed(RequestState from, RequestState to) noexcept {
  switch (from) {
  case RequestState::kWaiting:
    return to == RequestState::kPrefill || to == RequestState::kCancelled ||
           to == RequestState::kFailed;
  case RequestState::kPrefill:
    return to == RequestState::kDecoding || to == RequestState::kFinished ||
           to == RequestState::kCancelled || to == RequestState::kFailed;
  case RequestState::kDecoding:
    return to == RequestState::kFinished || to == RequestState::kCancelled ||
           to == RequestState::kFailed;
  case RequestState::kFinished:
  case RequestState::kCancelled:
  case RequestState::kFailed:
    return false;
  }
  return false;
}

Request::Request(RequestId id, std::vector<TokenId> prompt_tokens, std::size_t max_new_tokens,
                 RequestTimePoint created_at)
    : id_(id), prompt_tokens_(std::move(prompt_tokens)), max_new_tokens_(max_new_tokens),
      created_at_(created_at), latest_event_at_(created_at) {
  HELIX_CHECK(id_.valid(), ErrorCode::kInvalidArgument,
              "scheduler::Request: request ID zero is reserved as invalid");
  HELIX_CHECK(!prompt_tokens_.empty(), ErrorCode::kInvalidArgument,
              request_context(id_, "constructor") + ": prompt must contain at least one token");
}

Request::Request(Request&& other) noexcept
    : id_(other.id_), prompt_tokens_(std::move(other.prompt_tokens_)),
      generated_tokens_(std::move(other.generated_tokens_)), max_new_tokens_(other.max_new_tokens_),
      prompt_tokens_processed_(other.prompt_tokens_processed_), state_(other.state_),
      finish_reason_(other.finish_reason_), terminal_context_(std::move(other.terminal_context_)),
      created_at_(other.created_at_), latest_event_at_(other.latest_event_at_),
      first_scheduled_at_(other.first_scheduled_at_),
      prefill_completed_at_(other.prefill_completed_at_), first_token_at_(other.first_token_at_),
      last_token_at_(other.last_token_at_), completed_at_(other.completed_at_) {
  other.id_ = RequestId{};
  other.max_new_tokens_ = 0;
  other.prompt_tokens_processed_ = 0;
  other.state_ = RequestState::kFailed;
  other.finish_reason_ = FinishReason::kFailed;
  other.first_scheduled_at_.reset();
  other.prefill_completed_at_.reset();
  other.first_token_at_.reset();
  other.last_token_at_.reset();
  other.completed_at_.reset();
}

RequestId Request::id() const noexcept {
  return id_;
}

RequestState Request::state() const noexcept {
  return state_;
}

FinishReason Request::finish_reason() const noexcept {
  return finish_reason_;
}

bool Request::terminal() const noexcept {
  return request_state_is_terminal(state_);
}

std::span<const TokenId> Request::prompt_tokens() const noexcept {
  return prompt_tokens_;
}

std::span<const GeneratedToken> Request::generated_tokens() const noexcept {
  return generated_tokens_;
}

std::size_t Request::max_new_tokens() const noexcept {
  return max_new_tokens_;
}

std::size_t Request::prompt_tokens_processed() const noexcept {
  return prompt_tokens_processed_;
}

std::size_t Request::remaining_prompt_tokens() const noexcept {
  return prompt_tokens_.size() - prompt_tokens_processed_;
}

std::size_t Request::remaining_generation_tokens() const noexcept {
  return max_new_tokens_ - generated_tokens_.size();
}

std::string_view Request::terminal_context() const noexcept {
  return terminal_context_;
}

RequestTimePoint Request::created_at() const noexcept {
  return created_at_;
}

std::optional<RequestTimePoint> Request::first_scheduled_at() const noexcept {
  return first_scheduled_at_;
}

std::optional<RequestTimePoint> Request::prefill_completed_at() const noexcept {
  return prefill_completed_at_;
}

std::optional<RequestTimePoint> Request::first_token_at() const noexcept {
  return first_token_at_;
}

std::optional<RequestTimePoint> Request::last_token_at() const noexcept {
  return last_token_at_;
}

std::optional<RequestTimePoint> Request::completed_at() const noexcept {
  return completed_at_;
}

std::optional<RequestDuration> Request::waiting_time() const noexcept {
  if (!first_scheduled_at_.has_value()) {
    return std::nullopt;
  }
  return *first_scheduled_at_ - created_at_;
}

std::optional<RequestDuration> Request::time_to_first_token() const noexcept {
  if (!first_token_at_.has_value()) {
    return std::nullopt;
  }
  return *first_token_at_ - created_at_;
}

void Request::require_time_not_before_latest(RequestTimePoint now,
                                             std::string_view operation) const {
  HELIX_CHECK(now >= latest_event_at_, ErrorCode::kInvalidArgument,
              request_context(id_, operation) + ": event time precedes prior request activity");
}

void Request::transition_to(RequestState next, RequestTimePoint now, std::string_view operation) {
  require_time_not_before_latest(now, operation);
  if (!request_state_transition_allowed(state_, next)) {
    std::ostringstream message;
    message << request_context(id_, operation) << ": transition from " << request_state_name(state_)
            << " to " << request_state_name(next) << " is not allowed";
    throw_error(ErrorCode::kInvalidArgument, message.str());
  }
  state_ = next;
  latest_event_at_ = now;
}

void Request::begin_prefill(RequestTimePoint now) {
  transition_to(RequestState::kPrefill, now, "begin_prefill");
  first_scheduled_at_ = now;
}

void Request::record_prefill_progress(std::size_t token_count, RequestTimePoint now) {
  constexpr std::string_view kOperation = "record_prefill_progress";
  require_time_not_before_latest(now, kOperation);
  HELIX_CHECK(state_ == RequestState::kPrefill, ErrorCode::kInvalidArgument,
              request_context(id_, kOperation) + ": request is not in prefill");
  HELIX_CHECK(token_count != 0, ErrorCode::kInvalidArgument,
              request_context(id_, kOperation) + ": token count must be positive");
  HELIX_CHECK(token_count <= remaining_prompt_tokens(), ErrorCode::kInvalidArgument,
              request_context(id_, kOperation) + ": token count exceeds remaining prompt tokens");

  prompt_tokens_processed_ += token_count;
  latest_event_at_ = now;
  if (remaining_prompt_tokens() == 0) {
    prefill_completed_at_ = now;
    if (max_new_tokens_ == 0) {
      state_ = RequestState::kFinished;
      finish_reason_ = FinishReason::kLength;
      completed_at_ = now;
    } else {
      state_ = RequestState::kDecoding;
    }
  }
}

void Request::append_generated_token(TokenId token, RequestTimePoint now) {
  constexpr std::string_view kOperation = "append_generated_token";
  require_time_not_before_latest(now, kOperation);
  HELIX_CHECK(state_ == RequestState::kDecoding, ErrorCode::kInvalidArgument,
              request_context(id_, kOperation) + ": request is not decoding");
  HELIX_CHECK(generated_tokens_.size() < max_new_tokens_, ErrorCode::kInternal,
              request_context(id_, kOperation) + ": generation limit invariant was violated");

  generated_tokens_.push_back(GeneratedToken{token, now});
  if (!first_token_at_.has_value()) {
    first_token_at_ = now;
  }
  last_token_at_ = now;
  latest_event_at_ = now;
  if (generated_tokens_.size() == max_new_tokens_) {
    state_ = RequestState::kFinished;
    finish_reason_ = FinishReason::kLength;
    completed_at_ = now;
  }
}

void Request::finish_decoding(RequestTimePoint now) {
  constexpr std::string_view kOperation = "finish_decoding";
  transition_to(RequestState::kFinished, now, kOperation);
  finish_reason_ = FinishReason::kEndOfSequence;
  completed_at_ = now;
}

void Request::terminate(RequestState terminal_state, FinishReason reason, RequestTimePoint now,
                        std::string context, std::string_view operation) {
  HELIX_CHECK(!context.empty(), ErrorCode::kInvalidArgument,
              request_context(id_, operation) + ": terminal context must not be empty");
  transition_to(terminal_state, now, operation);
  terminal_context_ = std::move(context);
  finish_reason_ = reason;
  completed_at_ = now;
}

void Request::cancel(RequestTimePoint now, std::string context) {
  terminate(RequestState::kCancelled, FinishReason::kCancelled, now, std::move(context), "cancel");
}

void Request::fail(RequestTimePoint now, std::string context) {
  terminate(RequestState::kFailed, FinishReason::kFailed, now, std::move(context), "fail");
}

} // namespace helix::scheduler
