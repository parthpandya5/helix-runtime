#pragma once

#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace helix::scheduler {

class FifoScheduler;

class RequestId final {
public:
  RequestId() = default;
  explicit RequestId(std::uint64_t value) noexcept : value_(value) {}

  [[nodiscard]] bool valid() const noexcept {
    return value_ != 0;
  }
  [[nodiscard]] std::uint64_t value() const noexcept {
    return value_;
  }

  auto operator<=>(const RequestId&) const = default;

private:
  std::uint64_t value_ = 0;
};

class TokenId final {
public:
  TokenId() = default;
  explicit TokenId(std::uint32_t value) noexcept : value_(value) {}

  [[nodiscard]] std::uint32_t value() const noexcept {
    return value_;
  }

  auto operator<=>(const TokenId&) const = default;

private:
  std::uint32_t value_ = 0;
};

enum class RequestState : std::uint8_t {
  kWaiting,
  kPrefill,
  kDecoding,
  kFinished,
  kCancelled,
  kFailed,
};

enum class FinishReason : std::uint8_t {
  kNone,
  kLength,
  kEndOfSequence,
  kCancelled,
  kFailed,
};

[[nodiscard]] std::string_view request_state_name(RequestState state);
[[nodiscard]] std::string_view finish_reason_name(FinishReason reason);
[[nodiscard]] bool request_state_is_terminal(RequestState state) noexcept;
[[nodiscard]] bool request_state_transition_allowed(RequestState from, RequestState to) noexcept;

using RequestClock = std::chrono::steady_clock;
using RequestTimePoint = RequestClock::time_point;
using RequestDuration = RequestClock::duration;

struct GeneratedToken {
  TokenId id;
  RequestTimePoint emitted_at;

  friend bool operator==(const GeneratedToken&, const GeneratedToken&) = default;
};

// Mutable Request instances are owned by one scheduler thread and are not
// thread-safe. The type is move-only so a live request ID has one state owner.
class Request final {
public:
  Request(RequestId id, std::vector<TokenId> prompt_tokens, std::size_t max_new_tokens,
          RequestTimePoint created_at = RequestClock::now());

  Request(const Request&) = delete;
  Request& operator=(const Request&) = delete;
  Request(Request&& other) noexcept;
  Request& operator=(Request&&) = delete;

  [[nodiscard]] RequestId id() const noexcept;
  [[nodiscard]] RequestState state() const noexcept;
  [[nodiscard]] FinishReason finish_reason() const noexcept;
  [[nodiscard]] bool terminal() const noexcept;
  [[nodiscard]] std::span<const TokenId> prompt_tokens() const noexcept;
  [[nodiscard]] std::span<const GeneratedToken> generated_tokens() const noexcept;
  [[nodiscard]] std::size_t max_new_tokens() const noexcept;
  [[nodiscard]] std::size_t reused_prompt_tokens() const noexcept;
  [[nodiscard]] std::size_t prompt_tokens_processed() const noexcept;
  [[nodiscard]] std::size_t remaining_prompt_tokens() const noexcept;
  [[nodiscard]] std::size_t remaining_generation_tokens() const noexcept;
  [[nodiscard]] std::string_view terminal_context() const noexcept;

  [[nodiscard]] RequestTimePoint created_at() const noexcept;
  [[nodiscard]] RequestTimePoint latest_event_at() const noexcept;
  [[nodiscard]] std::optional<RequestTimePoint> first_scheduled_at() const noexcept;
  [[nodiscard]] std::optional<RequestTimePoint> prefill_completed_at() const noexcept;
  [[nodiscard]] std::optional<RequestTimePoint> first_token_at() const noexcept;
  [[nodiscard]] std::optional<RequestTimePoint> last_token_at() const noexcept;
  [[nodiscard]] std::optional<RequestTimePoint> completed_at() const noexcept;
  [[nodiscard]] std::optional<RequestDuration> waiting_time() const noexcept;
  [[nodiscard]] std::optional<RequestDuration> time_to_first_token() const noexcept;

  void begin_prefill(RequestTimePoint now);
  void record_prefill_progress(std::size_t token_count, RequestTimePoint now);
  void append_generated_token(TokenId token, RequestTimePoint now);
  void finish_decoding(RequestTimePoint now);
  void cancel(RequestTimePoint now, std::string context = "cancelled by caller");
  void fail(RequestTimePoint now, std::string context);

private:
  friend class FifoScheduler;

  void reserve_generated_capacity(std::size_t capacity);
  void seed_reused_prompt_tokens(std::size_t token_count) noexcept;
  void require_time_not_before_latest(RequestTimePoint now, std::string_view operation) const;
  void transition_to(RequestState next, RequestTimePoint now, std::string_view operation);
  void terminate(RequestState terminal_state, FinishReason reason, RequestTimePoint now,
                 std::string context, std::string_view operation);

  RequestId id_;
  std::vector<TokenId> prompt_tokens_;
  std::vector<GeneratedToken> generated_tokens_;
  std::size_t max_new_tokens_ = 0;
  std::size_t reused_prompt_tokens_ = 0;
  std::size_t prompt_tokens_processed_ = 0;
  RequestState state_ = RequestState::kWaiting;
  FinishReason finish_reason_ = FinishReason::kNone;
  std::string terminal_context_;
  RequestTimePoint created_at_;
  RequestTimePoint latest_event_at_;
  std::optional<RequestTimePoint> first_scheduled_at_;
  std::optional<RequestTimePoint> prefill_completed_at_;
  std::optional<RequestTimePoint> first_token_at_;
  std::optional<RequestTimePoint> last_token_at_;
  std::optional<RequestTimePoint> completed_at_;
};

} // namespace helix::scheduler
